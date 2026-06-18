import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from src.utils.batch_processor import BatchProcessor
from src.utils.config import load_config
from src.utils.constants import (
    LLMTypes,
    LLMModels,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_BATCH_SIZE,
    DATA_DIR
)
from src.new_classifier.experiment_manager import (
    build_prompt_snapshot,
    experiment_paths,
    finalize_experiment_metadata,
    initialize_experiment_metadata,
    resolve_experiment_dir,
)
from src.utils.category_parser import (
    build_category_lookup,
    parse_category_response,
    parse_classification_response,
)
from src.utils.gemini_classifier import GeminiClassifier
from src.utils.jsonl_utils import read_jsonl, append_jsonl, update_jsonl
from src.utils.message_utils import create_classification_messages

# Defaults for the original new-classifier run (unchanged)
DEFAULT_CATEGORIES_FILE = Path(__file__).parent / "categories.json"
DEFAULT_FEW_SHOT_FILE = (
    Path(__file__).parent
    / "cohere-embeddings__agglomerative_k=8__gemini-3.1-pro-preview.json"
)
DEFAULT_OUTPUT_FILE = DATA_DIR / "new_classification_results.csv"
DEFAULT_BATCH_STATUS_FILE = DATA_DIR / "new_batch_status.jsonl"
DEFAULT_BATCH_DIR = "new_batches"

INPUT_FILE = DATA_DIR / "classification_results.csv"
TARGET_OLD_CATEGORY = "אמינה כי בהירה, פשוטה, עקבית, אותנטית, ללא הגזמה"
GEMINI_SAVE_EVERY = 10
DEFAULT_GEMINI_WORKERS = 5
RESULT_COLUMNS = [
    "sentence_id", "origin_sentence", "category", "new_category", "model_reasoning",
]
ERROR_CATEGORIES = {"", "Classification Error"}


def parse_args():
    parser = argparse.ArgumentParser(description='Classify sentences with new categories')
    parser.add_argument(
        '--categories-file',
        type=str,
        default=str(DEFAULT_CATEGORIES_FILE),
        help='Categories file (.json or .csv with cluster,label columns)',
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output CSV path (auto-derived from categories file if omitted)',
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'gemini'],
        default='openai',
        help='LLM provider (default: openai)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name (defaults: gpt-4o for openai, gemini-3.5-flash for gemini)',
    )
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help='Maximum tokens for completion')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help='Temperature for sampling')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for OpenAI batch processing')
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_GEMINI_WORKERS,
        help='Number of parallel Gemini API workers (use 1 for sequential)',
    )
    parser.add_argument('--check_batches', action='store_true', default=True,
                        help='Check status of previously submitted OpenAI batches')
    parser.add_argument(
        '--input-scope',
        type=str,
        choices=['target_category', 'all'],
        default='target_category',
        help=(
            'Which sentences to classify: '
            '"target_category" = only "אמינה כי בהירה..." (default, ~441); '
            '"all" = every unique sentence in classification_results.csv (~1,626)'
        ),
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Ignore saved output file and classify all input sentences from scratch',
    )
    parser.add_argument(
        '--retry-errors',
        action='store_true',
        help='Re-classify rows previously saved with an empty or error category',
    )
    parser.add_argument(
        '--experiment',
        action='store_true',
        help='Save results under data/experiments/expN/ with metadata.json',
    )
    parser.add_argument(
        '--new-experiment',
        action='store_true',
        help='Create a new expN folder (implies --experiment; skips match prompt)',
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help=(
            'Implies --experiment. Skip the terminal question when a matching '
            'experiment is found; automatically resume from that folder'
        ),
    )
    parser.add_argument(
        '--few-shot-per-category',
        type=int,
        default=0,
        metavar='N',
        help=(
            'Number of curated few-shot examples per category from the curation '
            f'JSON (default: 0 = no few-shot). File: {DEFAULT_FEW_SHOT_FILE.name}'
        ),
    )
    parser.add_argument(
        '--few-shot-file',
        type=str,
        default=str(DEFAULT_FEW_SHOT_FILE),
        help='Path to curated few-shot JSON (used when --few-shot-per-category > 0)',
    )
    parser.add_argument(
        '--print-prompt',
        action='store_true',
        help='Print the full classification prompt for one sentence and exit (no API call)',
    )
    parser.add_argument(
        '--shuffle-examples',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Shuffle few-shot (sentence, category) pairs across categories before '
            'building the prompt (default: True). Use --no-shuffle-examples to keep '
            'grouped order by category.'
        ),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for few-shot shuffle (default: 42)',
    )
    return parser.parse_args()


def normalize_args(args):
    """Flags that only make sense with experiment folders."""
    if args.new_experiment or args.non_interactive:
        args.experiment = True
    return args


def resolve_paths(categories_file: Path, output_file: Optional[str]) -> Tuple[Path, Path, str]:
    """Derive output, batch status, and batch dir from categories file."""
    if output_file:
        output_path = Path(output_file)
        stem = output_path.stem.replace("_classification_results", "")
    elif categories_file.name == DEFAULT_CATEGORIES_FILE.name:
        output_path = DEFAULT_OUTPUT_FILE
        stem = "new"
    else:
        stem = categories_file.stem
        output_path = DATA_DIR / f"{stem}_classification_results.csv"

    if stem == "new":
        batch_status_file = DEFAULT_BATCH_STATUS_FILE
        batch_dir_name = DEFAULT_BATCH_DIR
    else:
        batch_status_file = DATA_DIR / f"{stem}_batch_status.jsonl"
        batch_dir_name = f"{stem}_batches"

    return output_path, batch_status_file, batch_dir_name


def load_categories(path: Path) -> Dict:
    if path.suffix.lower() == ".csv":
        return load_categories_csv(path)
    return load_categories_json(path)


def load_categories_json(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_categories_csv(path: Path) -> Dict:
    df = pd.read_csv(path)
    categories = []
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        categories.append({
            "id": int(row["cluster"]) + 1,
            "name": label,
            "description": label,
        })
    return {"categories": categories, "few_shot_examples": []}


def format_categories_text(categories_data: Dict) -> str:
    categories_text = "קטגוריות לסיווג:\n\n"

    for cat in categories_data["categories"]:
        categories_text += f"{cat['id']}: {cat['name']}\n"
        # categories_text += f"תיאור: {cat['description']}\n"
        categories_text += "\n"

    return categories_text


def create_examples_df(categories_data: Dict) -> pd.DataFrame:
    examples = []
    for example in categories_data.get("few_shot_examples", []):
        examples.append({
            "sentence": example["statement"],
            "category": f"{example['category_id']}: {example['category_name']}"
        })
    return pd.DataFrame(examples)


def _find_curation_entry(
    curation_categories: Dict,
    category_id: int,
    category_name: str,
) -> Optional[Dict]:
    for entry in curation_categories.values():
        if entry.get("cluster_display") == category_id:
            return entry
    for entry in curation_categories.values():
        if entry.get("label", "").strip() == category_name.strip():
            return entry
    return None


def load_few_shot_from_curation(
    curation_path: Path,
    categories_data: Dict,
    per_category: int,
) -> pd.DataFrame:
    if per_category <= 0:
        return pd.DataFrame(columns=["sentence", "category"])

    with open(curation_path, "r", encoding="utf-8") as f:
        curation_data = json.load(f)

    curation_categories = curation_data.get("categories", {})
    examples = []

    for cat in categories_data["categories"]:
        cat_id = cat["id"]
        cat_name = cat["name"]
        entry = _find_curation_entry(curation_categories, cat_id, cat_name)
        if entry is None:
            print(
                f"Warning: no curation entry for category {cat_id}: {cat_name[:60]}..."
            )
            continue

        best_sentences = entry.get("best_sentences", [])
        if not best_sentences:
            print(f"Warning: no best_sentences for category {cat_id}: {cat_name[:60]}...")
            continue

        for sentence in best_sentences[:per_category]:
            examples.append({
                "sentence": sentence,
                "category": f"{cat_id}: {cat_name}",
            })

    return pd.DataFrame(examples)


def shuffle_examples_df(
    examples_df: pd.DataFrame,
    shuffle: bool,
    seed: int,
) -> pd.DataFrame:
    if examples_df.empty or not shuffle:
        return examples_df
    return examples_df.sample(frac=1, random_state=seed).reset_index(drop=True)


def build_examples_df(
    categories_data: Dict,
    few_shot_per_category: int = 0,
    few_shot_file: Optional[Path] = None,
    shuffle_examples: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    if few_shot_per_category > 0:
        path = few_shot_file or DEFAULT_FEW_SHOT_FILE
        examples_df = load_few_shot_from_curation(
            path, categories_data, few_shot_per_category
        )
    else:
        examples_df = create_examples_df(categories_data)

    return shuffle_examples_df(examples_df, shuffle_examples, seed)


def print_prompt_preview(
    sentence: str,
    categories_text: str,
    examples_df: pd.DataFrame,
) -> None:
    use_few_shot = examples_df is not None and not examples_df.empty
    messages = create_classification_messages(
        sentence=sentence,
        categories_text=categories_text,
        examples_df=examples_df if use_few_shot else None,
    )
    print("=== SYSTEM PROMPT ===")
    print(messages[0]["content"])
    print("\n=== USER PROMPT ===")
    print(messages[1]["content"])
    if use_few_shot:
        print(f"\n=== FEW-SHOT SUMMARY ({len(examples_df)} examples) ===")
        for _, row in examples_df.iterrows():
            preview = row["sentence"][:80] + ("..." if len(row["sentence"]) > 80 else "")
            print(f"  [{row['category']}] {preview}")


def chunkify(items: List, chunk_size: int) -> List[List]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def load_input_sentences(input_scope: str = "target_category") -> pd.DataFrame:
    sentences_df = pd.read_csv(INPUT_FILE)

    if input_scope == "all":
        return sentences_df.drop_duplicates(subset=["sentence_id", "title"])

    # Legacy behavior: dedup full file first, then keep target old category only
    sentences_df = sentences_df.drop_duplicates(subset=["sentence_id", "title"])
    return sentences_df[sentences_df["category"] == TARGET_OLD_CATEGORY]


def load_cached_results(output_file: Path) -> pd.DataFrame:
    if not output_file.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)

    cached_df = pd.read_csv(output_file)
    for col in RESULT_COLUMNS:
        if col not in cached_df.columns:
            cached_df[col] = pd.NA
    cached_df["sentence_id"] = cached_df["sentence_id"].astype(str)
    before = len(cached_df)
    cached_df = cached_df.drop_duplicates(subset=["sentence_id"], keep="last")

    if len(cached_df) != before:
        cached_df.to_csv(output_file, index=False)
        print(f"Normalized cache: removed {before - len(cached_df)} duplicate rows in {output_file}")

    return cached_df


def get_cached_sentence_ids(cached_df: pd.DataFrame, retry_errors: bool) -> set:
    if cached_df.empty:
        return set()

    if not retry_errors:
        return set(cached_df["sentence_id"].astype(str))

    retry_mask = (
        cached_df["new_category"].isna()
        | cached_df["new_category"].astype(str).str.strip().isin(ERROR_CATEGORIES)
    )
    valid_df = cached_df[~retry_mask]
    return set(valid_df["sentence_id"].astype(str))


def save_cached_results(results_df: pd.DataFrame, output_file: Path) -> pd.DataFrame:
    results_df = results_df.copy()
    results_df["sentence_id"] = results_df["sentence_id"].astype(str)
    results_df = results_df.drop_duplicates(subset=["sentence_id"], keep="last")
    results_df.to_csv(output_file, index=False)
    return results_df


def prepare_sentences_list(unclassified_df: pd.DataFrame) -> List[Dict]:
    sentences_list = []
    for _, row in unclassified_df.iterrows():
        data = row.to_dict()
        data['sentence_id'] = str(row['sentence_id'])
        sentences_list.append(data)
    return sentences_list


def save_result_row(
    results_df: pd.DataFrame,
    sentence_id: str,
    origin_sentence: str,
    old_category: str,
    new_category: str,
    model_reasoning: Optional[str] = None,
) -> pd.DataFrame:
    new_row = pd.DataFrame([{
        "sentence_id": sentence_id,
        "origin_sentence": origin_sentence,
        "category": old_category,
        "new_category": new_category,
        "model_reasoning": model_reasoning,
    }])
    return pd.concat([results_df, new_row], ignore_index=True)


def classify_with_gemini(
    classifier: GeminiClassifier,
    sentences_list: List[Dict],
    cached_results: pd.DataFrame,
    categories_text: str,
    examples_df: pd.DataFrame,
    output_file: Path,
    valid_names: set,
    id_to_name: Dict[int, str],
    workers: int = DEFAULT_GEMINI_WORKERS,
) -> pd.DataFrame:
    results_df = cached_results.copy()
    total = len(sentences_list)
    use_few_shot = examples_df is not None and not examples_df.empty
    examples = examples_df if use_few_shot else None
    lock = threading.Lock()
    completed = 0

    def classify_one(sentence_data: Dict) -> Dict:
        sentence_id = sentence_data["sentence_id"]
        try:
            raw_response = classifier.classify_sentence(
                sentence=sentence_data["origin_sentence"],
                categories_text=categories_text,
                examples_df=examples,
            )
            new_category, model_reasoning = parse_classification_response(
                raw_response, id_to_name, valid_names
            )
        except Exception as e:
            print(f"Unexpected error on sentence {sentence_id}: {e}")
            new_category = "Classification Error"
            model_reasoning = None

        return {
            "sentence_id": sentence_id,
            "origin_sentence": sentence_data["origin_sentence"],
            "category": sentence_data["category"],
            "new_category": new_category,
            "model_reasoning": model_reasoning,
        }

    def handle_result(row: Dict) -> None:
        nonlocal results_df, completed
        with lock:
            results_df = save_result_row(
                results_df,
                sentence_id=row["sentence_id"],
                origin_sentence=row["origin_sentence"],
                old_category=row["category"],
                new_category=row["new_category"],
                model_reasoning=row.get("model_reasoning"),
            )
            completed += 1
            if completed % GEMINI_SAVE_EVERY == 0 or completed == total:
                results_df = save_cached_results(results_df, output_file)
                print(
                    f"Gemini progress: {completed}/{total} new, "
                    f"{len(results_df)} total cached in {output_file}"
                )

    if workers <= 1:
        for sentence_data in sentences_list:
            handle_result(classify_one(sentence_data))
    else:
        print(f"Running Gemini with {workers} parallel workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(classify_one, s) for s in sentences_list]
            for future in as_completed(futures):
                handle_result(future.result())

    print(f"Gemini classification complete: {len(results_df)} total rows in {output_file}")
    return results_df


def run_openai_batches(
    args,
    sentences_list: List[Dict],
    sentences_df: pd.DataFrame,
    categories_text: str,
    examples_df: pd.DataFrame,
    output_file: Path,
    batch_status_file: Path,
    batch_dir_name: str,
    valid_names: set,
    id_to_name: Dict[int, str],
) -> None:
    batch_processor = BatchProcessor(
        api_key=load_config(LLMTypes.OPENAI),
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_dir_name=batch_dir_name,
    )

    all_batches = {}
    if batch_status_file.exists():
        for r in read_jsonl(batch_status_file):
            all_batches[r['batch_id']] = r

    if args.check_batches:
        completed_results = []

        for batch_id, batch_info in list(all_batches.items()):
            if batch_info['status'] in ['completed', 'failed', 'expired', 'cancelled']:
                continue

            try:
                status = batch_processor.get_batch_status(batch_id)
                print(f"Batch {batch_id}: {status}")

                batch_info['status'] = status
                update_jsonl(batch_status_file, batch_info, 'batch_id')

                if status == 'completed':
                    results = batch_processor.get_batch_results(batch_id)
                    completed_results.extend(results)
                    print(f"Retrieved {len(results)} results from batch {batch_id}")
            except Exception as e:
                print(f"Error checking batch {batch_id}: {e}")

        if completed_results:
            results_df = load_cached_results(output_file)

            new_results = []
            for result in completed_results:
                sentence_id = result['custom_id'].split('request-')[1]
                raw_category = result['category']
                new_category, model_reasoning = parse_classification_response(
                    raw_category, id_to_name, valid_names
                )

                matching_rows = sentences_df[sentences_df['sentence_id'].astype(str) == sentence_id]
                if len(matching_rows) > 0:
                    sentence_row = matching_rows.iloc[0]
                    new_results.append({
                        "sentence_id": sentence_id,
                        "origin_sentence": sentence_row['origin_sentence'],
                        "category": sentence_row['category'],
                        "new_category": new_category,
                        "model_reasoning": model_reasoning,
                    })

            if new_results:
                new_results_df = pd.DataFrame(new_results)
                results_df = pd.concat([results_df, new_results_df], ignore_index=True)
                results_df = save_cached_results(results_df, output_file)
                print(f"Saved {len(new_results)} new results to {output_file}")

                classified_ids = get_cached_sentence_ids(results_df, args.retry_errors)
                remaining = sentences_df[~sentences_df["sentence_id"].astype(str).isin(classified_ids)]
                sentences_list = prepare_sentences_list(remaining)

    if len(sentences_list) == 0:
        print("No more sentences to classify")
        return

    chunks = chunkify(sentences_list, args.batch_size)

    for chunk_data in chunks:
        start_idx = chunk_data[0]['sentence_id']
        end_idx = chunk_data[-1]['sentence_id']
        chunk_range = f"{start_idx}-{end_idx}"

        for info in all_batches.values():
            if info['index_range'] == chunk_range and info['status'] not in [
                "failed", "expired", "cancelled", "completed"
            ]:
                print(f"Already have a pending batch for sentences {chunk_range}")
                break
        else:
            batch_file = batch_processor.prepare_or_get_batch_file(
                chunk_data,
                categories_text,
                examples_df if not examples_df.empty else None,
            )
            new_batch_id = batch_processor.submit_batch(batch_file)

            record = {
                'batch_id': new_batch_id,
                'index_range': chunk_range,
                'total_sentences': len(chunk_data),
                'status': 'pending',
            }
            append_jsonl(batch_status_file, record)
            all_batches[new_batch_id] = record
            print(f"Submitted batch {new_batch_id} for sentences {chunk_range}")

    print("All new batches submitted. Run again later to check status.")


def main():
    args = normalize_args(parse_args())

    categories_file = Path(args.categories_file)

    if args.model is None:
        args.model = LLMModels.GEMINI if args.provider == "gemini" else LLMModels.OPENAI

    categories_data = load_categories(categories_file)
    categories_text = format_categories_text(categories_data)
    few_shot_file = Path(args.few_shot_file)
    examples_df = build_examples_df(
        categories_data,
        few_shot_per_category=args.few_shot_per_category,
        few_shot_file=few_shot_file,
        shuffle_examples=args.shuffle_examples,
        seed=args.seed,
    )
    id_to_name, valid_names = build_category_lookup(categories_data["categories"])
    prompt_snapshot = build_prompt_snapshot(categories_text, examples_df)

    exp_dir = None
    if args.experiment:
        exp_dir = resolve_experiment_dir(
            provider=args.provider,
            model=args.model,
            prompt_snapshot=prompt_snapshot,
            force_new=args.new_experiment,
            interactive=not args.non_interactive,
        )
        paths = experiment_paths(exp_dir)
        output_file = paths["results"]
        batch_status_file = paths["batch_status"]
        batch_dir_name = paths["batch_dir_name"]
        initialize_experiment_metadata(
            exp_dir=exp_dir,
            provider=args.provider,
            model=args.model,
            categories_file=categories_file,
            input_scope=args.input_scope,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            prompt_snapshot=prompt_snapshot,
        )
    else:
        output_file, batch_status_file, batch_dir_name = resolve_paths(
            categories_file, args.output_file
        )

    print(f"Categories: {categories_file} ({len(categories_data['categories'])} labels)")
    if exp_dir:
        print(f"Experiment: {exp_dir}")
    print(f"Output: {output_file}")
    print(f"Provider: {args.provider} ({args.model})")
    print(f"Input scope: {args.input_scope}")
    if args.few_shot_per_category > 0:
        shuffle_note = (
            f"shuffled (seed={args.seed})"
            if args.shuffle_examples
            else "grouped by category"
        )
        print(
            f"Few-shot: {args.few_shot_per_category} per category "
            f"({len(examples_df)} total, {shuffle_note}) from {few_shot_file}"
        )
    else:
        print("Few-shot: disabled (0 per category)")

    sentences_df = load_input_sentences(args.input_scope)
    print(f"Loaded {len(sentences_df)} sentences from {INPUT_FILE}")

    cached_results = pd.DataFrame(columns=RESULT_COLUMNS)
    if not args.no_cache:
        cached_results = load_cached_results(output_file)
        if not cached_results.empty:
            print(f"Cache hit: {len(cached_results)} results in {output_file}")
    else:
        print("Cache disabled (--no-cache)")

    classified_ids = get_cached_sentence_ids(cached_results, args.retry_errors)
    unclassified_df = sentences_df[~sentences_df["sentence_id"].astype(str).isin(classified_ids)]

    if len(unclassified_df) == 0:
        print("All sentences have been classified (nothing left in cache to do)")
        if exp_dir:
            finalize_experiment_metadata(exp_dir, output_file)
        return

    print(
        f"Resume: {len(classified_ids)} cached, "
        f"{len(unclassified_df)} remaining to classify"
    )
    sentences_list = prepare_sentences_list(unclassified_df)

    if args.print_prompt:
        sample = sentences_list[0] if sentences_list else {
            "origin_sentence": "(no input sentences — using placeholder)",
        }
        print_prompt_preview(
            sentence=sample["origin_sentence"],
            categories_text=categories_text,
            examples_df=examples_df,
        )
        return

    if args.provider == "gemini":
        classifier = GeminiClassifier(
            api_key=load_config(LLMTypes.GEMINI),
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        classify_with_gemini(
            classifier=classifier,
            sentences_list=sentences_list,
            cached_results=cached_results,
            categories_text=categories_text,
            examples_df=examples_df,
            output_file=output_file,
            valid_names=valid_names,
            id_to_name=id_to_name,
            workers=args.workers,
        )
    else:
        run_openai_batches(
            args=args,
            sentences_list=sentences_list,
            sentences_df=sentences_df,
            categories_text=categories_text,
            examples_df=examples_df,
            output_file=output_file,
            batch_status_file=batch_status_file,
            batch_dir_name=batch_dir_name,
            valid_names=valid_names,
            id_to_name=id_to_name,
        )

    if exp_dir:
        finalize_experiment_metadata(exp_dir, output_file)
        print(f"Experiment metadata saved to {exp_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
