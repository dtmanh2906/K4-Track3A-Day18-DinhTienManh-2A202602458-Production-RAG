"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")


def is_valid_key(key: str) -> bool:
    if not key or "..." in key or "key_that" in key or key.startswith("sk-..."):
        return False
    return True


def get_llm_client():
    """Returns (client, model_name) configured for either Google Gemini or OpenAI."""
    if is_valid_key(GOOGLE_API_KEY):
        from openai import OpenAI
        client = OpenAI(
            api_key=GOOGLE_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries=1,
            timeout=15.0,
        )
        return client, "gemini-flash-latest"
    elif is_valid_key(OPENAI_API_KEY):
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY, max_retries=1, timeout=15.0), "gpt-4o-mini"
    return None, None


try:
    import transformers.utils.import_utils as _tiu
    _tiu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
