import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Default Constants
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "hr_policies"
VECTOR_DB_DIR = BASE_DIR / "vectorstore"

# API Keys & LLM settings
DEFAULT_GROQ_KEY = "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"

def get_groq_api_key(override_key: str = None) -> str:
    """Retrieve Groq API Key from override, env, or default fallback."""
    if override_key and override_key.strip():
        return override_key.strip()
    
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
        
    return DEFAULT_GROQ_KEY

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "llama3-8b-8192")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunking settings
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 4
