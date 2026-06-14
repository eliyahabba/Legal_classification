import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.utils.config import load_config
from src.utils.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    EXPERIMENTS_DIR,
    LLMModels,
    LLMTypes,
)
from src.utils.gemini_classifier import GeminiClassifier

PROMPT_FILE = (
    Path(__file__).parent.parent / "utils" / "lying_woman_myth_prompt.txt"
)

ORIGINAL_COLUMNS = ["sentence_id", "origin_sentence", "category", "new_category"]
MYTH_COLUMNS = [
    "invokes_myth",
    "myth_polarity",
    "myth_trigger",
    "myth_reason",
    "myth_raw",
]
RESULT_COLUMNS = ORIGINAL_COLUMNS + MYTH_COLUMNS

SAVE_EVERY = 10
DEFAULT_WORKERS = 5


def experiment_paths(exp: str) -> Tuple[Path, Path]:
    exp_dir = EXPERIMENTS_DIR / exp
    return (
        exp_dir / "classification_results.csv",
        exp_dir / "classification_results_with_myth.csv",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label sentences with lying-woman myth annotations"
    )
    parser.add_argument(
        "exp",
        help="Experiment folder name (e.g. exp7)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Smoke test: classify first row only, print output, do not write CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLMModels.GEMINI,
        help=f"Gemini model (default: {LLMModels.GEMINI})",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens for completion",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel Gemini API workers (use 1 for sequential)",
    )
    return parser.parse_args()


def load_prompt_template() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(template: str, sentence: str) -> str:
    return template.replace("{SENTENCE}", sentence)


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_myth_response(raw: str) -> Tuple[Dict[str, Optional[object]], bool]:
    """
    Parse model JSON into myth columns.
    Returns (field_dict, parsed_ok).
    """
    empty = {
        "invokes_myth": None,
        "myth_polarity": None,
        "myth_trigger": None,
        "myth_reason": None,
    }
    if not raw or not str(raw).strip():
        return empty, False

    try:
        data = json.loads(_extract_json_text(str(raw)))
    except (json.JSONDecodeError, TypeError):
        return empty, False

    if not isinstance(data, dict):
        return empty, False

    trigger = data.get("trigger")
    if trigger is None or (isinstance(trigger, str) and trigger.lower() == "null"):
        trigger = None

    invokes_myth = data.get("invokes_myth")
    if isinstance(invokes_myth, str):
        invokes_myth = invokes_myth.strip().lower() in ("true", "1", "yes")

    return {
        "invokes_myth": invokes_myth,
        "myth_polarity": data.get("polarity"),
        "myth_trigger": trigger,
        "myth_reason": data.get("reason"),
    }, True


def has_myth_label(row: pd.Series) -> bool:
    raw = row.get("myth_raw")
    return pd.notna(raw) and str(raw).strip() != ""


def load_results(input_df: pd.DataFrame, output_file: Path) -> pd.DataFrame:
    if not output_file.exists():
        results_df = input_df.copy()
        for col in MYTH_COLUMNS:
            results_df[col] = pd.NA
        return results_df

    cached_df = pd.read_csv(output_file)
    cached_df["sentence_id"] = cached_df["sentence_id"].astype(str)
    input_df = input_df.copy()
    input_df["sentence_id"] = input_df["sentence_id"].astype(str)

    results_df = input_df.merge(
        cached_df[["sentence_id"] + MYTH_COLUMNS],
        on="sentence_id",
        how="left",
    )

    return results_df[RESULT_COLUMNS]


def save_results(results_df: pd.DataFrame, output_file: Path) -> None:
    results_df = results_df.copy()
    results_df["sentence_id"] = results_df["sentence_id"].astype(str)
    results_df.to_csv(output_file, index=False)


def apply_myth_fields(row: Dict, raw: str) -> Dict:
    parsed, ok = parse_myth_response(raw)
    row["myth_raw"] = raw
    if ok:
        row.update(parsed)
    else:
        row["invokes_myth"] = None
        row["myth_polarity"] = None
        row["myth_trigger"] = None
        row["myth_reason"] = None
    return row


def run_test(classifier: GeminiClassifier, template: str, row: pd.Series) -> None:
    prompt = build_prompt(template, row["origin_sentence"])
    raw = classifier.generate(prompt)
    parsed, ok = parse_myth_response(raw)

    print("=== sentence_id ===")
    print(row["sentence_id"])
    print("\n=== raw model output ===")
    print(raw)
    print("\n=== parsed fields ===")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    print(f"\n=== parse ok: {ok} ===")

    required = ["invokes_myth", "myth_polarity", "myth_trigger", "myth_reason", "myth_raw"]
    row_out = apply_myth_fields(row.to_dict(), raw)
    missing = [col for col in required if col not in row_out]
    if missing:
        print(f"Missing fields: {missing}")
    elif not ok:
        print("JSON parsing failed — fix parser or prompt before full run.")
    else:
        print("Smoke test passed: all five myth fields present and JSON parsed.")


def classify_rows(
    classifier: GeminiClassifier,
    template: str,
    results_df: pd.DataFrame,
    to_process: pd.DataFrame,
    output_file: Path,
    workers: int,
) -> pd.DataFrame:
    total = len(to_process)
    if total == 0:
        return results_df

    lock = threading.Lock()
    completed = 0
    results_by_id = results_df.set_index("sentence_id", drop=False).to_dict("index")

    def classify_one(row: pd.Series) -> Tuple[str, Dict]:
        sentence_id = str(row["sentence_id"])
        raw = classifier.generate(build_prompt(template, row["origin_sentence"]))
        updated = apply_myth_fields(row.to_dict(), raw)
        return sentence_id, updated

    def handle_result(sentence_id: str, updated: Dict) -> None:
        nonlocal completed
        with lock:
            results_by_id[sentence_id] = updated
            completed += 1
            if completed % SAVE_EVERY == 0 or completed == total:
                save_results(
                    pd.DataFrame(list(results_by_id.values()))[RESULT_COLUMNS],
                    output_file,
                )
                print(f"Progress: {completed}/{total} new, saved to {output_file}")

    if workers <= 1:
        for _, row in to_process.iterrows():
            sentence_id, updated = classify_one(row)
            handle_result(sentence_id, updated)
    else:
        print(f"Running with {workers} parallel workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(classify_one, row): row
                for _, row in to_process.iterrows()
            }
            for future in as_completed(futures):
                sentence_id, updated = future.result()
                handle_result(sentence_id, updated)

    return pd.DataFrame(list(results_by_id.values()))[RESULT_COLUMNS]


def print_summary(results_df: pd.DataFrame) -> None:
    labeled = results_df[results_df["myth_raw"].notna() & (results_df["myth_raw"].astype(str).str.strip() != "")]
    print(f"\nSummary: {len(results_df)} total rows, {len(labeled)} labeled")
    if labeled.empty:
        return
    counts = labeled["myth_polarity"].value_counts(dropna=False)
    print("myth_polarity counts:")
    for value, count in counts.items():
        print(f"  {value}: {count}")


def main():
    args = parse_args()
    input_file, output_file = experiment_paths(args.exp)
    if not input_file.exists():
        raise FileNotFoundError(f"Input not found: {input_file}")

    template = load_prompt_template()
    input_df = pd.read_csv(input_file)
    input_df["sentence_id"] = input_df["sentence_id"].astype(str)

    classifier = GeminiClassifier(
        api_key=load_config(LLMTypes.GEMINI),
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.test:
        run_test(classifier, template, input_df.iloc[0])
        return

    results_df = load_results(input_df, output_file)
    to_process = results_df[~results_df.apply(has_myth_label, axis=1)]

    print(f"Experiment: {args.exp}")
    print(f"Input: {input_file} ({len(input_df)} rows)")
    print(f"Output: {output_file}")
    print(f"Model: {args.model}")
    print(f"Resume: {len(input_df) - len(to_process)} cached, {len(to_process)} remaining")

    if len(to_process) == 0:
        print("All rows already labeled.")
        print_summary(results_df)
        return

    results_df = classify_rows(
        classifier=classifier,
        template=template,
        results_df=results_df,
        to_process=to_process,
        output_file=output_file,
        workers=args.workers,
    )
    save_results(results_df, output_file)
    print_summary(results_df)


if __name__ == "__main__":
    main()
