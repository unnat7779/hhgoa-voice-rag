import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
QDRANT_STORAGE_DIR = BASE_DIR / "qdrant_storage"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Voice / STT Configuration
SARVAM_MODEL = "saaras:v3"
SARVAM_STT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"

# LLM Models on Groq Cloud
GROQ_MODEL_FAST = "allam-2-7b"
GROQ_MODEL_SMART = "qwen/qwen3.6-27b"

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Reranker Model Configuration
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Qdrant Vector DB Settings
QDRANT_PATH = str(QDRANT_STORAGE_DIR)
COLLECTION_SEMANTIC = "msmarco_semantic_chunks"
COLLECTION_PASSAGE = "msmarco_passage_chunks"
COLLECTION_SENTENCE = "msmarco_sentence_chunks"
COLLECTION_HIERARCHICAL = "msmarco_hierarchical_chunks"

# Chunking Configuration
SEMANTIC_SIMILARITY_THRESHOLD = 0.72
SENTENCE_WINDOW_SIZE = 1  # ±1 context window
RECURSIVE_CHUNK_SIZE = 256  # tokens / approx words
RECURSIVE_CHUNK_OVERLAP = 50

# Retrieval & Fusion Parameters
TOP_K_RETRIEVAL = 8
TOP_K_RERANK = 4
RRF_K_CONSTANT = 60
METADATA_BOOST_SELECTED = 1.25

# Guardrails Parameters
GUARDRAIL_SIMILARITY_THRESHOLD = 0.50
GUARDRAIL_GROUNDING_THRESHOLD = 0.45
BLOCKED_TOPICS = [
    "weapon", "bomb", "hate speech", "malware", "exploit", "hack bank",
    "credit card fraud", "suicide instruction", "terrorist"
]

# Dataset Configuration
DEFAULT_LANGUAGE = "hi"  # Hindi partition of MSMARCO-XI
MAX_PASSAGES_INDEX = int(os.getenv("MAX_PASSAGES_INDEX", "25000"))
