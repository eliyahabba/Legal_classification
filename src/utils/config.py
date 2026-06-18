import os

from dotenv import load_dotenv

from src.utils.constants import LLMTypes


def load_config(model_type: LLMTypes):
    """
    Load configuration from .env file
    """
    load_dotenv()

    if model_type == LLMTypes.OPENAI:
        api_key = os.getenv('OPENAI_KEY') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_KEY or OPENAI_API_KEY must be set in .env file")
    elif model_type == LLMTypes.GEMINI:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in .env file")
    else:  # claude
        if not os.getenv('ANTHROPIC_API_KEY'):
            raise ValueError("ANTHROPIC_API_KEY must be set in .env file")
        api_key = os.getenv('ANTHROPIC_API_KEY')
    return api_key
