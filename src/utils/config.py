import os

from dotenv import load_dotenv

from src.utils.constants import LLMTypes


def load_config(model_type: LLMTypes):
    """
    Load configuration from .env file
    """
    load_dotenv()

    if model_type == LLMTypes.OPENAI:
        if not os.getenv('OPENAI_KEY'):
            raise ValueError("OPENAI_KEY must be set in .env file")
        api_key = os.getenv('OPENAI_KEY')
    else:  # claude
        if not os.getenv('ANTHROPIC_API_KEY'):
            raise ValueError("ANTHROPIC_API_KEY must be set in .env file")
        api_key = os.getenv('ANTHROPIC_API_KEY')
    return api_key
