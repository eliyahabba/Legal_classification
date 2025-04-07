import json
from typing import List, Dict, Optional

import pandas as pd
from openai import OpenAI

from src.utils.message_utils import create_classification_messages
from src.utils.constants import DATA_DIR


class BatchProcessor:
    def __init__(self,
                 api_key: str,
                 model: str,
                 max_tokens: int,
                 temperature: float):
        self.client = OpenAI(api_key=api_key)
        self.batch_dir = DATA_DIR / "batches"
        self.batch_dir.mkdir(exist_ok=True)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def get_existing_batch_file(self, start_idx: int, end_idx: int) -> Optional[str]:
        """Check if a batch file already exists for the given index range"""
        batch_file = self.batch_dir / f"batch_input_{start_idx}_to_{end_idx}.jsonl"
        return str(batch_file) if batch_file.exists() else None

    def prepare_or_get_batch_file(self,
                                 sentences: List[Dict],
                                 categories_text: str,
                                 examples_df: Optional[pd.DataFrame] = None) -> str:
        """Get existing batch file or prepare a new one"""
        start_idx = sentences[0].get('index', 0)
        end_idx = sentences[-1].get('index', len(sentences)-1)
        
        # Check if batch file already exists
        existing_file = self.get_existing_batch_file(start_idx, end_idx)
        if existing_file:
            print(f"Found existing batch file for indices {start_idx} to {end_idx}")
            return existing_file
        
        # If not, prepare new batch file
        batch_file = self.batch_dir / f"batch_input_{start_idx}_to_{end_idx}.jsonl"
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            for i, sentence_data in enumerate(sentences):
                messages = create_classification_messages(
                    sentence=sentence_data['origin_sentence'],
                    categories_text=categories_text,
                    examples_df=examples_df
                )
                
                request = {
                    "custom_id": f"request-{sentence_data.get('index', i)}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }
                }
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        
        return str(batch_file)

    def submit_batch(self, input_file: str) -> str:
        """Submit batch job to OpenAI"""
        # Upload the file
        with open(input_file, 'rb') as f:
            file = self.client.files.create(file=f, purpose="batch")

        # Create the batch
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        return batch.id

    def get_batch_status(self, batch_id: str) -> str:
        """Get status of a batch job"""
        batch = self.client.batches.retrieve(batch_id)
        return batch.status

    def get_batch_results(self, batch_id: str) -> List[Dict]:
        """Get results from a completed batch"""
        batch = self.client.batches.retrieve(batch_id)
        if batch.status != "completed":
            raise ValueError(f"Batch {batch_id} is not completed (status: {batch.status})")

        # Get the output file
        file_response = self.client.files.content(batch.output_file_id)
        results = []

        for line in file_response.text.split('\n'):
            if line.strip():
                result = json.loads(line)
                if result['response']['status_code'] == 200:
                    category = result['response']['body']['choices'][0]['message']['content']
                    results.append({
                        'custom_id': result['custom_id'],
                        'category': category.strip()
                    })
                else:
                    results.append({
                        'custom_id': result['custom_id'],
                        'category': 'Classification Error'
                    })

        return results
