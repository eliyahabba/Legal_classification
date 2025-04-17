import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

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
from src.utils.jsonl_utils import read_jsonl, append_jsonl, update_jsonl
from src.utils.llm import LLM

# Constants specific to new classifier
CATEGORIES_JSON_FILE = Path(__file__).parent / "categories.json"
INPUT_FILE = DATA_DIR / "classification_results.csv"
OUTPUT_FILE = DATA_DIR / "new_classification_results.csv"
BATCH_STATUS_FILE = DATA_DIR / "new_batch_status.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description='Classify sentences with new categories')
    parser.add_argument('--model', type=str, default=LLMModels.OPENAI,
                        help='LLM model to use')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help='Maximum tokens for completion')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help='Temperature for sampling')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for processing')
    parser.add_argument('--check_batches', action='store_true', default=True,
                        help='Check status of previously submitted batches')
    return parser.parse_args()


def load_categories_json() -> Dict:
    """Load categories and examples from JSON file"""
    with open(CATEGORIES_JSON_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_categories_text(categories_data: Dict) -> str:
    """Format categories into text for prompt"""
    categories_text = "קטגוריות לסיווג:\n\n"
    
    for cat in categories_data["categories"]:
        categories_text += f"{cat['id']}: {cat['name']}\n"
        categories_text += f"תיאור: {cat['description']}\n"
        # if "keywords" in cat and cat["keywords"]:
        #     categories_text += f"מילות מפתח: {', '.join(cat['keywords'])}\n"
        categories_text += "\n"
    
    return categories_text


def create_examples_df(categories_data: Dict) -> pd.DataFrame:
    """Create examples dataframe from categories data"""
    examples = []
    for example in categories_data.get("few_shot_examples", []):
        examples.append({
            "sentence": example["statement"],
            "category": f"{example['category_id']}: {example['category_name']}"
        })
    return pd.DataFrame(examples)


def chunkify(items: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size"""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def clean_category(category: str) -> str:
    """Clean up category text from model response"""
    # Remove any extra text or formatting
    category = category.strip()
    
    # Handle common formatting issues
    if ":" in category:
        # If the model returns "Category X: Name", extract just the name
        parts = category.split(":", 1)
        if len(parts) > 1:
            category = parts[1].strip()
    
    return category


def main():
    args = parse_args()
    
    # Load categories and format them for prompts
    categories_data = load_categories_json()
    categories_text = format_categories_text(categories_data)
    examples_df = create_examples_df(categories_data)
    
    # Load input data
    sentences_df = pd.read_csv(INPUT_FILE)
    # filter only "אמינה כי בהירה, פשוטה, עקבית, אותנטית, ללא הגזמה" category
    sentences_df = sentences_df.drop_duplicates(subset=["sentence_id", "title"])
    sentences_df = sentences_df[sentences_df["category"] == "אמינה כי בהירה, פשוטה, עקבית, אותנטית, ללא הגזמה"]
    # drop duplicates

    print(f"Loaded {len(sentences_df)} sentences from {INPUT_FILE}")
    
    # Load existing results if file exists
    existing_results = pd.DataFrame()
    if OUTPUT_FILE.exists():
        existing_results = pd.read_csv(OUTPUT_FILE)
        print(f"Loaded {len(existing_results)} existing classifications")
    
    # Find sentences that haven't been classified yet
    if not existing_results.empty:
        classified_ids = set(existing_results["sentence_id"].astype(str))
        unclassified_df = sentences_df[~sentences_df["sentence_id"].astype(str).isin(classified_ids)]
    else:
        unclassified_df = sentences_df
    
    if len(unclassified_df) == 0:
        print("All sentences have been classified")
        return
    
    print(f"Found {len(unclassified_df)} sentences to classify")
    
    # Prepare all unclassified sentences
    sentences_list = []
    for idx, row in unclassified_df.iterrows():
        data = row.to_dict()
        data['sentence_id'] = str(row['sentence_id'])
        sentences_list.append(data)
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        api_key=load_config(LLMTypes.OPENAI),
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_dir_name="new_batches"
    )
    
    # Load existing batch statuses
    all_batches = {}
    if BATCH_STATUS_FILE.exists():
        records = read_jsonl(BATCH_STATUS_FILE)
        for r in records:
            batch_id = r['batch_id']
            all_batches[batch_id] = r
    
    # Check batch statuses if requested
    if args.check_batches:
        completed_results = []
        
        for batch_id, batch_info in list(all_batches.items()):
            if batch_info['status'] in ['completed', 'failed', 'expired', 'cancelled']:
                continue
            
            try:
                status = batch_processor.get_batch_status(batch_id)
                print(f"Batch {batch_id}: {status}")
                
                batch_info['status'] = status
                update_jsonl(BATCH_STATUS_FILE, batch_info, 'batch_id')
                
                if status == 'completed':
                    results = batch_processor.get_batch_results(batch_id)
                    completed_results.extend(results)
                    print(f"Retrieved {len(results)} results from batch {batch_id}")
            except Exception as e:
                print(f"Error checking batch {batch_id}: {e}")
        
        # Process completed results
        if completed_results:
            # Load or create results dataframe
            if OUTPUT_FILE.exists():
                results_df = pd.read_csv(OUTPUT_FILE)
            else:
                results_df = pd.DataFrame(columns=["sentence_id", "origin_sentence", "category", "new_category"])
            
            # Process each result
            new_results = []
            for result in completed_results:
                sentence_id = result['custom_id'].split('request-')[1]
                category = clean_category(result['category'])
                
                # Find the original sentence in the dataset
                matching_rows = sentences_df[sentences_df['sentence_id'].astype(str) == sentence_id]
                if len(matching_rows) > 0:
                    sentence_row = matching_rows.iloc[0]
                    
                    # Add to results
                    new_results.append({
                        "sentence_id": sentence_id,
                        "origin_sentence": sentence_row['origin_sentence'],
                        "category": sentence_row['category'],
                        "new_category": category
                    })
            
            # Add new results to dataframe
            if new_results:
                new_results_df = pd.DataFrame(new_results)
                results_df = pd.concat([results_df, new_results_df], ignore_index=True)
                results_df.to_csv(OUTPUT_FILE, index=False)
                print(f"Saved {len(new_results)} new results to {OUTPUT_FILE}")
            
            # Update unclassified sentences
            if new_results:
                classified_ids = set(results_df["sentence_id"].astype(str))
                unclassified_df = sentences_df[~sentences_df["sentence_id"].astype(str).isin(classified_ids)]
                
                # Update sentences list
                sentences_list = []
                for idx, row in unclassified_df.iterrows():
                    data = row.to_dict()
                    data['sentence_id'] = str(row['sentence_id'])
                    sentences_list.append(data)
    
    # If no sentences left to classify, exit
    if len(sentences_list) == 0:
        print("No more sentences to classify")
        return
    
    # Split into chunks based on batch size
    chunks = chunkify(sentences_list, args.batch_size)
    
    # Create batches for each chunk
    for chunk_data in chunks:
        start_idx = chunk_data[0]['sentence_id']
        end_idx = chunk_data[-1]['sentence_id']
        chunk_range = f"{start_idx}-{end_idx}"
        
        # Check if we already have a batch for this range
        for b_id, info in all_batches.items():
            if info['index_range'] == chunk_range and info['status'] not in ["failed", "expired", "cancelled", "completed"]:
                print(f"Already have a pending batch for sentences {chunk_range}")
                break
        else:
            # Submit new batch
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
            print(f"Submitted batch {new_batch_id} for sentences {chunk_range}")
    
    print("All new batches submitted. Run again later to check status.")


if __name__ == "__main__":
    main() 