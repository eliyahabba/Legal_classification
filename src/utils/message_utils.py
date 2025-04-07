from typing import List, Dict, Optional
import pandas as pd

from src.utils.prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    SENTENCE_CLASSIFICATION_PROMPT,
    FEW_SHOT_CLASSIFICATION_PROMPT_BEFORE,
    FEW_SHOT_CLASSIFICATION_PROMPT_AFTER
)

def create_few_shot_examples(examples_df: pd.DataFrame) -> str:
    """Create few-shot examples string from example data"""
    examples = []
    for _, row in examples_df.iterrows():
        example = f"משפט: \"{row['sentence']}\"\nקטגוריה: {row['category']}\n"
        examples.append(example)
    return "\n".join(examples)

def create_classification_messages(
        sentence: str,
        categories_text: str,
        examples_df: Optional[pd.DataFrame] = None
) -> List[Dict]:
    """Create messages for classification request"""
    if examples_df is not None:
        messages = [
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": FEW_SHOT_CLASSIFICATION_PROMPT_BEFORE.format(
                    categories=categories_text
                ) +
                create_few_shot_examples(examples_df) +
                FEW_SHOT_CLASSIFICATION_PROMPT_AFTER.format(
                    sentence=sentence
                )
            }
        ]
    else:
        messages = [
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SENTENCE_CLASSIFICATION_PROMPT.format(
                    categories=categories_text,
                    sentence=sentence
                )
            }
        ]
    return messages 