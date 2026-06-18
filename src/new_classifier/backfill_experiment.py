"""
One-time helper: migrate an existing flat results CSV into an experiment folder
with full metadata (prompt, model, date).
"""
import argparse
from pathlib import Path

from src.new_classifier.classify_dataset import (
    create_examples_df,
    format_categories_text,
    load_categories,
)
from src.new_classifier.experiment_manager import (
    backfill_experiment_from_csv,
    build_prompt_snapshot,
)
from src.utils.constants import (
    DATA_DIR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    LLMModels,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill an experiment folder from an existing results CSV"
    )
    parser.add_argument(
        "--source-csv",
        type=str,
        default=str(
            DATA_DIR / "categories_agglomerative_k8_classification_results.csv"
        ),
        help="Existing flat results CSV to migrate",
    )
    parser.add_argument(
        "--categories-file",
        type=str,
        default="src/new_classifier/categories_agglomerative_k8.csv",
    )
    parser.add_argument("--provider", type=str, default="gemini")
    parser.add_argument("--model", type=str, default=LLMModels.GEMINI)
    parser.add_argument("--input-scope", type=str, default="all")
    parser.add_argument("--date", type=str, default="2026-06-10",
                        help="Run date (YYYY-MM-DD, day only)")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="exp1",
        help="Target experiment folder name (default: exp1)",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args()


def main():
    args = parse_args()

    source_csv = Path(args.source_csv)
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    categories_file = Path(args.categories_file)
    categories_data = load_categories(categories_file)
    categories_text = format_categories_text(categories_data)
    examples_df = create_examples_df(categories_data)
    prompt_snapshot = build_prompt_snapshot(categories_text, examples_df)

    exp_dir = backfill_experiment_from_csv(
        source_csv=source_csv,
        provider=args.provider,
        model=args.model,
        categories_file=categories_file,
        input_scope=args.input_scope,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        prompt_snapshot=prompt_snapshot,
        run_date=args.date,
        experiment_name=args.experiment_name,
    )

    print(f"Backfill complete: {exp_dir}")
    print(f"  Results: {exp_dir / 'classification_results.csv'}")
    print(f"  Metadata: {exp_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
