import argparse
import re
from typing import List, Optional, Dict

import pandas as pd

from src.utils.batch_processor import BatchProcessor
from src.utils.config import load_config
from src.utils.constants import (
    LLMTypes,
    SENTENCES_FILE,
    CATEGORIES_FILE,
    EXAMPLES_FILE,
    RESULTS_FILE,
    LLMModels,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_BATCH_SIZE,
    BATCH_STATUS_FILE
)
from src.utils.jsonl_utils import read_jsonl, append_jsonl, update_jsonl
from src.utils.llm import LLM
from src.utils.message_utils import create_classification_messages


def parse_args():
    parser = argparse.ArgumentParser(description='Classify sentences using OpenAI Batch API')
    parser.add_argument('--model', type=str, default=LLMModels.OPENAI,
                        help='OpenAI model to use')
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help='Maximum tokens for completion')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help='Temperature for completion')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Number of sentences per batch file')
    parser.add_argument('--few-shot', action='store_true',
                        help='Use few-shot examples if specified, otherwise zero-shot')
    return parser.parse_args()


def clean_category(category: str) -> str:
    """Remove leading number, period, and space from category string."""
    # Check if the category starts with a number followed by period and space
    if re.match(r'^\d+\.\s', category):
        return re.sub(r'^\d+\.\s', '', category)
    return category


# Use the function to clean the category

def read_categories(file_path: str) -> List[str]:
    """Read categories from a CSV file"""
    df = pd.read_csv(file_path)
    return df['קטגוריה מפורטת'].tolist()


def format_categories(categories: List[str]) -> str:
    """Format categories list with numbers"""
    return "\n".join([f"{i + 1}. {cat}" for i, cat in enumerate(categories)])


def chunkify(lst: List[Dict], size: int) -> List[List[Dict]]:
    """Split a list into chunks of at most 'size'."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def classify_sentence(
        sentence: str,
        categories: List[str],
        llm: LLM,
        examples: Optional[pd.DataFrame] = None
) -> str:
    """Classify a single sentence using the LLM"""
    try:
        categories_text = format_categories(categories)
        messages = create_classification_messages(sentence, categories_text, examples)

        response = llm.generate(
            messages=messages,
            system=None  # System message is now included in messages list
        )

        return str(response).strip()

    except Exception as e:
        print(f"Error classifying sentence: {sentence}")
        print(f"Error: {str(e)}")
        return "Classification Error"


def main():
    args = parse_args()

    # Read input data
    sentences_df = pd.read_csv(SENTENCES_FILE)
    categories = read_categories(CATEGORIES_FILE)
    categories_text = format_categories(categories)

    # Get unique sentences
    unique_sentences_df = sentences_df.drop_duplicates(subset=['origin_sentence'], keep='first')
    print(f"Found {len(unique_sentences_df)} unique sentences out of {len(sentences_df)} total")

    # Load existing results if file exists
    existing_results = pd.DataFrame()
    if RESULTS_FILE.exists():
        existing_results = pd.read_csv(RESULTS_FILE)
        print(f"Loaded {len(existing_results)} existing classifications")

    # Find sentences that haven't been classified yet by take unique_sentences_df and remove existing_results
    if not existing_results.empty:
        merged_df = unique_sentences_df.merge(
            existing_results[['origin_sentence', 'category']],
            on='origin_sentence',
            how='left'
        )
        unclassified_df = merged_df[merged_df['category'].isna()]
        # drop the category column
        unclassified_df = unclassified_df.drop(columns=['category'])
    else:
        unclassified_df = unique_sentences_df

    if len(unclassified_df) == 0:
        print("All sentences have been classified")
        return

    print(f"Found {len(unclassified_df)} sentences to classify")

    # Prepare all unclassified sentences (store original index)
    sentences_list = []
    for idx, row in unclassified_df.iterrows():
        data = row.to_dict()
        data['index'] = idx
        sentences_list.append(data)

    # Initialize batch processor
    batch_processor = BatchProcessor(
        api_key=load_config(LLMTypes.OPENAI),
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    # Load existing batch statuses
    all_batches = {}
    if BATCH_STATUS_FILE.exists():
        records = read_jsonl(BATCH_STATUS_FILE)
        for r in records:
            batch_id = r['batch_id']
            all_batches[batch_id] = r

    # Iterate over existing batches, check statuses
    completed_results = []
    for batch_id, info in list(all_batches.items()):
        status = batch_processor.get_batch_status(batch_id)
        print(f"Batch {batch_id} status: {status}")

        if status == "completed":
            batch_res = batch_processor.get_batch_results(batch_id)
            completed_results.extend(batch_res)
            # Update status in jsonl
            updated_record = {
                'batch_id': batch_id,
                'index_range': info['index_range'],
                'total_sentences': info['total_sentences'],
                'status': 'completed'
            }
            update_jsonl(BATCH_STATUS_FILE, updated_record, 'batch_id')
            del all_batches[batch_id]
        elif status in ["failed", "expired", "cancelled"]:
            print(f"Batch {batch_id} {status}, we can re-run or handle that specifically")
            updated_record = {
                'batch_id': batch_id,
                'index_range': info['index_range'],
                'total_sentences': info['total_sentences'],
                'status': status
            }
            update_jsonl(BATCH_STATUS_FILE, updated_record, 'batch_id')
            del all_batches[batch_id]
        else:
            print(f"Batch {batch_id} is still {status}")
            updated_record = {
                'batch_id': batch_id,
                'index_range': info['index_range'],
                'total_sentences': info['total_sentences'],
                'status': status
            }
            update_jsonl(BATCH_STATUS_FILE, updated_record, 'batch_id')

    # Process completed results
    if completed_results:
        new_results = []
        for result in completed_results:
            idx_str = result['custom_id'].split('-')[1]
            original_idx = int(idx_str)
            if original_idx in unclassified_df.index:
                sentence_data = unclassified_df.loc[original_idx].to_dict()
                sentence_data['category'] = clean_category(result['category'])
                if sentence_data['category'] not in categories:
                    print(f"Invalid category: {result['category']}")
                new_results.append(sentence_data)
            elif original_idx in unclassified_df.sentence_id.values:
                sentence_data = unclassified_df[unclassified_df['sentence_id'] == original_idx].iloc[0].to_dict()
                sentence_data['category'] = clean_category(result['category'])
                if sentence_data['category'] not in categories:
                    print(f"Invalid category: {result['category']}")
                new_results.append(sentence_data)

        new_results_df = pd.DataFrame(new_results)
        final_results_df = pd.concat([existing_results, new_results_df], ignore_index=True)
        final_results_df.to_csv(RESULTS_FILE, index=False, encoding='utf-8-sig')
        print(f"Added {len(new_results)} new classifications")

    # Determine unclassified sentences
    if len(completed_results) > 0:
        existing_results = pd.read_csv(RESULTS_FILE)
        merged_df2 = unclassified_df.merge(
            existing_results[['origin_sentence', 'category']],
            on='origin_sentence',
            how='left'
        )
        unclassified_df = merged_df2[merged_df2['category'].isna()]

    if len(unclassified_df) == 0:
        print("No more unclassified sentences after updating completed batches.")
        return

    # Split into chunks based on batch size
    big_sentences_list = []
    for idx, row in unclassified_df.iterrows():
        d = row.to_dict()
        d['index'] = row['sentence_id']
        big_sentences_list.append(d)

    chunks = chunkify(big_sentences_list, args.batch_size)
    if not chunks:
        print("No new chunks to process")
        return

    # Load few-shot examples if specified
    examples_df = None
    if args.few_shot and EXAMPLES_FILE.exists():
        examples_df = pd.read_csv(EXAMPLES_FILE)

    # Create batches for each chunk
    for chunk_data in chunks:
        start_idx = chunk_data[0]['index']
        end_idx = chunk_data[-1]['index']
        chunk_range = f"{start_idx}-{end_idx}"

        for b_id, info in all_batches.items():
            if info['index_range'] == chunk_range and info['status'] not in ["failed", "expired", "cancelled",
                                                                             "completed"]:
                print(f"Already have a pending batch for sentences {chunk_range}")
                break
        else:
            batch_file = batch_processor.prepare_or_get_batch_file(
                chunk_data,
                categories_text,
                examples_df
            )
            new_batch_id = batch_processor.submit_batch(batch_file)

            record = {
                'batch_id': new_batch_id,
                'index_range': chunk_range,
                'total_sentences': len(chunk_data),
                'status': 'pending'
            }
            append_jsonl(BATCH_STATUS_FILE, record)
            all_batches[new_batch_id] = record

    print("All new batches submitted. Run again later to check status.")


if __name__ == "__main__":
    main()
