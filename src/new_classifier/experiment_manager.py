import hashlib
import json
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.constants import DATA_DIR, EXPERIMENTS_DIR
from src.utils.message_utils import create_few_shot_examples
from src.utils.prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    FEW_SHOT_CLASSIFICATION_PROMPT_AFTER,
    FEW_SHOT_CLASSIFICATION_PROMPT_BEFORE,
    SENTENCE_CLASSIFICATION_PROMPT,
)

EXPERIMENT_RESULTS_NAME = "classification_results.csv"
EXPERIMENT_METADATA_NAME = "metadata.json"
EXPERIMENT_BATCH_STATUS_NAME = "batch_status.jsonl"
SENTENCE_PLACEHOLDER = "<SENTENCE>"


def build_prompt_snapshot(categories_text: str, examples_df: pd.DataFrame) -> Dict[str, Any]:
    use_few_shot = examples_df is not None and not examples_df.empty

    if use_few_shot:
        user_prompt_template = (
            FEW_SHOT_CLASSIFICATION_PROMPT_BEFORE.format(categories=categories_text)
            + create_few_shot_examples(examples_df)
            + FEW_SHOT_CLASSIFICATION_PROMPT_AFTER.format(sentence=SENTENCE_PLACEHOLDER)
        )
    else:
        user_prompt_template = SENTENCE_CLASSIFICATION_PROMPT.format(
            categories=categories_text,
            sentence=SENTENCE_PLACEHOLDER,
        )

    return {
        "system_prompt": CLASSIFICATION_SYSTEM_PROMPT,
        "user_prompt_template": user_prompt_template,
        "few_shot": use_few_shot,
        "sentence_placeholder": SENTENCE_PLACEHOLDER,
        "categories_text": categories_text,
    }


def build_match_key(provider: str, model: str, prompt_snapshot: Dict[str, Any]) -> str:
    payload = "\n---\n".join([
        provider,
        model,
        prompt_snapshot["system_prompt"],
        prompt_snapshot["user_prompt_template"],
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def list_experiment_dirs() -> List[Path]:
    if not EXPERIMENTS_DIR.exists():
        return []

    exp_dirs = []
    for path in EXPERIMENTS_DIR.iterdir():
        if path.is_dir() and re.fullmatch(r"exp\d+", path.name):
            exp_dirs.append(path)

    return sorted(exp_dirs, key=lambda p: int(p.name.replace("exp", "")))


def next_experiment_dir() -> Path:
    existing = list_experiment_dirs()
    if not existing:
        exp_dir = EXPERIMENTS_DIR / "exp1"
    else:
        last_num = int(existing[-1].name.replace("exp", ""))
        exp_dir = EXPERIMENTS_DIR / f"exp{last_num + 1}"

    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def load_experiment_metadata(exp_dir: Path) -> Optional[Dict[str, Any]]:
    metadata_path = exp_dir / EXPERIMENT_METADATA_NAME
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_matching_experiment(match_key: str) -> Optional[Path]:
    for exp_dir in list_experiment_dirs():
        metadata = load_experiment_metadata(exp_dir)
        if metadata and metadata.get("match_key") == match_key:
            return exp_dir
    return None


def count_cached_results(exp_dir: Path) -> int:
    results_path = exp_dir / EXPERIMENT_RESULTS_NAME
    if not results_path.exists():
        return 0
    return len(pd.read_csv(results_path))


def prompt_reuse_experiment(exp_dir: Path, metadata: Dict[str, Any]) -> bool:
    cached_rows = count_cached_results(exp_dir)
    print("\nMatching experiment found:")
    print(f"  Folder: {exp_dir.name}")
    print(f"  Model:  {metadata.get('provider')} / {metadata.get('model')}")
    print(f"  Date:   {metadata.get('date')}")
    print(f"  Cached: {cached_rows} rows in {EXPERIMENT_RESULTS_NAME}")
    print()

    while True:
        answer = input("Use this experiment folder and resume from cache? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no", ""):
            return False
        print("Please answer y or n.")


def build_experiment_metadata(
    experiment_id: str,
    provider: str,
    model: str,
    categories_file: Path,
    input_scope: str,
    temperature: float,
    max_tokens: int,
    prompt_snapshot: Dict[str, Any],
    match_key: str,
    run_date: Optional[str] = None,
    result_count: Optional[int] = None,
) -> Dict[str, Any]:
    metadata = {
        "experiment_id": experiment_id,
        "date": run_date or date.today().isoformat(),
        "provider": provider,
        "model": model,
        "categories_file": str(categories_file),
        "input_scope": input_scope,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "match_key": match_key,
        "match_params": {
            "provider": provider,
            "model": model,
        },
        "prompt": prompt_snapshot,
        "result_count": result_count,
    }
    return metadata


def save_experiment_metadata(exp_dir: Path, metadata: Dict[str, Any]) -> Path:
    exp_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = exp_dir / EXPERIMENT_METADATA_NAME
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata_path


def resolve_experiment_dir(
    provider: str,
    model: str,
    prompt_snapshot: Dict[str, Any],
    force_new: bool = False,
    interactive: bool = True,
) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    match_key = build_match_key(provider, model, prompt_snapshot)

    if not force_new:
        matching = find_matching_experiment(match_key)
        if matching:
            metadata = load_experiment_metadata(matching)
            if metadata:
                reuse = True
                if interactive:
                    reuse = prompt_reuse_experiment(matching, metadata)
                if reuse:
                    print(f"Resuming experiment in {matching}")
                    return matching

    exp_path = next_experiment_dir()
    print(f"Created new experiment folder: {exp_path}")
    return exp_path


def experiment_paths(exp_dir: Path) -> Dict[str, Path]:
    batch_dir_name = f"experiments/{exp_dir.name}/batches"
    return {
        "results": exp_dir / EXPERIMENT_RESULTS_NAME,
        "metadata": exp_dir / EXPERIMENT_METADATA_NAME,
        "batch_status": exp_dir / EXPERIMENT_BATCH_STATUS_NAME,
        "batch_dir_name": batch_dir_name,
    }


def initialize_experiment_metadata(
    exp_dir: Path,
    provider: str,
    model: str,
    categories_file: Path,
    input_scope: str,
    temperature: float,
    max_tokens: int,
    prompt_snapshot: Dict[str, Any],
    run_date: Optional[str] = None,
) -> Dict[str, Any]:
    existing = load_experiment_metadata(exp_dir)
    if existing:
        existing["result_count"] = count_cached_results(exp_dir)
        save_experiment_metadata(exp_dir, existing)
        return existing

    match_key = build_match_key(provider, model, prompt_snapshot)
    metadata = build_experiment_metadata(
        experiment_id=exp_dir.name,
        provider=provider,
        model=model,
        categories_file=categories_file,
        input_scope=input_scope,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_snapshot=prompt_snapshot,
        match_key=match_key,
        run_date=run_date,
        result_count=count_cached_results(exp_dir),
    )
    save_experiment_metadata(exp_dir, metadata)
    return metadata


def finalize_experiment_metadata(exp_dir: Path, results_file: Path) -> None:
    metadata = load_experiment_metadata(exp_dir)
    if not metadata:
        return

    metadata["result_count"] = count_cached_results(exp_dir) if results_file.exists() else 0
    if not metadata.get("date"):
        metadata["date"] = date.today().isoformat()
    save_experiment_metadata(exp_dir, metadata)


def backfill_experiment_from_csv(
    source_csv: Path,
    provider: str,
    model: str,
    categories_file: Path,
    input_scope: str,
    temperature: float,
    max_tokens: int,
    prompt_snapshot: Dict[str, Any],
    run_date: str,
    experiment_name: Optional[str] = None,
) -> Path:
    """Copy an existing flat results CSV into an experiment folder with metadata."""
    if experiment_name:
        exp_dir = EXPERIMENTS_DIR / experiment_name
    else:
        exp_dir = EXPERIMENTS_DIR / "exp1" if not list_experiment_dirs() else next_experiment_dir()

    exp_dir.mkdir(parents=True, exist_ok=True)
    paths = experiment_paths(exp_dir)

    shutil.copy2(source_csv, paths["results"])

    metadata = initialize_experiment_metadata(
        exp_dir=exp_dir,
        provider=provider,
        model=model,
        categories_file=categories_file,
        input_scope=input_scope,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_snapshot=prompt_snapshot,
        run_date=run_date,
    )
    finalize_experiment_metadata(exp_dir, paths["results"])
    return exp_dir
