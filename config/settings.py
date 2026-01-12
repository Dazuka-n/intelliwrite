import os
import warnings
from dotenv import load_dotenv

load_dotenv()


SUPPORTED_MODELS = {
    "gemini-flash": "models/gemini-flash-latest",
    "gemini-pro": "models/gemini-pro-latest",
    "gemini-1.5-flash": "models/gemini-flash-latest",
    "gemini-1.5-pro": "models/gemini-pro-latest",
}


def _normalize_gemini_model(model_name: str) -> str:
    """Map deprecated Gemini model IDs to currently supported ones."""
    if not model_name:
        return "models/gemini-flash-latest"

    normalized = model_name.strip()
    normalized_lower = normalized.lower()

    if normalized_lower.startswith("models/"):
        return normalized

    replacement = SUPPORTED_MODELS.get(normalized_lower)
    if replacement:
        warnings.warn(
            f"MODEL_NAME '{model_name}' is deprecated or unsupported for the current Gemini API. "
            f"Using '{replacement}' instead.",
            RuntimeWarning,
        )
        return replacement

    # Default to a safe, supported model name
    return "models/gemini-flash-latest"


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "aeo_knowledge_base")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/aeo_blog_db")

    # Gemini configuration (single provider)
    GEMINI_API_KEY = os.getenv("API")
    MODEL_NAME = _normalize_gemini_model(os.getenv("MODEL_NAME", "models/gemini-flash-latest"))

    DEFAULT_LLM_PROVIDER = "google"
    DEFAULT_LLM_MODEL = MODEL_NAME
    DEFAULT_LLM_API_KEY = GEMINI_API_KEY

    # Agent-specific Configurations (all use Gemini now)
    RESEARCHER_PROVIDER = DEFAULT_LLM_PROVIDER
    RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", DEFAULT_LLM_MODEL)
    RESEARCHER_API_KEY = DEFAULT_LLM_API_KEY

    PLANNER_PROVIDER = DEFAULT_LLM_PROVIDER
    PLANNER_MODEL = os.getenv("PLANNER_MODEL", DEFAULT_LLM_MODEL)
    PLANNER_API_KEY = DEFAULT_LLM_API_KEY

    WRITER_PROVIDER = DEFAULT_LLM_PROVIDER
    WRITER_MODEL = os.getenv("WRITER_MODEL", DEFAULT_LLM_MODEL)
    WRITER_API_KEY = DEFAULT_LLM_API_KEY

    OPTIMIZER_PROVIDER = DEFAULT_LLM_PROVIDER
    OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", DEFAULT_LLM_MODEL)
    OPTIMIZER_API_KEY = DEFAULT_LLM_API_KEY

    QA_PROVIDER = DEFAULT_LLM_PROVIDER
    QA_MODEL = os.getenv("QA_MODEL", DEFAULT_LLM_MODEL)
    QA_API_KEY = DEFAULT_LLM_API_KEY
