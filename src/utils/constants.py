from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SCHEMAS_DIR = DATA_DIR / "schemas"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

# Data files
SENTENCES_FILE = DATA_DIR / "sentences.csv"
CATEGORIES_FILE = DATA_DIR / "categories.csv"
EXAMPLES_FILE = DATA_DIR / "examples.csv"
RESULTS_FILE = DATA_DIR / "classification_results.csv"

# LLM Types
class LLMTypes:
    OPENAI = "openai"

# LLM Models
class LLMModels:
    OPENAI = "gpt-4o"

# Processing parameters
TEMPERATURE = 0.2






# Default OpenAI parameters
DEFAULT_MAX_TOKENS = 2500
DEFAULT_TEMPERATURE = 0.2
DEFAULT_BATCH_SIZE = 50

# Batch processing
BATCH_STATUS_FILE = DATA_DIR / "batch_status.jsonl"
BATCH_RESULTS_FILE = DATA_DIR / "batch_results.csv"
