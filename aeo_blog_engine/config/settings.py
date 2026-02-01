import os
import warnings
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)


SUPPORTED_MODELS = {
    "gemini-flash": "models/gemini-flash-latest",
    "gemini-pro": "models/gemini-pro-latest",
    "gemini-1.5-flash": "models/gemini-flash-latest",
    "gemini-1.5-pro": "models/gemini-pro-latest",
}

OPENROUTER_MODEL_ALIASES = {
    "models/gemini-flash-latest": "google/gemini-flash-1.5",
    "models/gemini-pro-latest": "google/gemini-pro-1.5",
}


def _normalize_non_google_model(model_name: str) -> str:
    if not model_name:
        return None
    return OPENROUTER_MODEL_ALIASES.get(model_name, model_name)


def _normalize_gemini_model(model_name: str) -> str:
    """Map deprecated Gemini model IDs to currently supported ones."""
    if not model_name:
        return "models/gemini-flash-latest"

    normalized = model_name.strip()
    normalized_lower = normalized.lower()

    if normalized_lower.startswith("models/gemini"):
        return normalized

    replacement = SUPPORTED_MODELS.get(normalized_lower)
    if replacement:
        return replacement

    # If it's not a known gemini model, default to gemini-flash
    if not normalized_lower.startswith("models/"):
         # Check if it's just the name without prefix
         if normalized_lower in SUPPORTED_MODELS:
             return SUPPORTED_MODELS[normalized_lower]
             
    warnings.warn(
        f"Model '{model_name}' is not a recognized Gemini model. "
        f"Defaulting to 'models/gemini-flash-latest'.",
        RuntimeWarning,
    )
    return "models/gemini-flash-latest"


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "aeo_knowledge_base")
    
    # Database Configuration with Auto-Correction for pg8000
    _raw_db_url = os.getenv("DATABASE_URL", "postgresql+pg8000://user:password@localhost:5432/aeo_blog_db")
    
    # 1. Force driver to pg8000
    if _raw_db_url.startswith("postgres://"):
        _url_with_driver = _raw_db_url.replace("postgres://", "postgresql+pg8000://", 1)
    elif _raw_db_url.startswith("postgresql://"):
        _url_with_driver = _raw_db_url.replace("postgresql://", "postgresql+pg8000://", 1)
    else:
        _url_with_driver = _raw_db_url

    # 2. Strip incompatible query parameters (sslmode, channel_binding) for pg8000
    # pg8000 uses create_engine(connect_args={'ssl_context': ...}) instead of URL params
    if "?" in _url_with_driver:
        # Split and keep only the base URL part
        DATABASE_URL = _url_with_driver.split("?")[0]
    else:
        DATABASE_URL = _url_with_driver
        
    # Debug Logging for Vercel
    print(f"--- DB CONFIG ---")
    print(f"Raw URL start: {_raw_db_url[:15]}...")
    print(f"Final URL start: {DATABASE_URL[:25]}...")
    print(f"Driver check: {'pg8000' in DATABASE_URL}")
    print(f"-----------------")

    # Gemini configuration (using OpenAI compatibility layer)
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # Prioritize keys that are known to be Gemini keys in this project
    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY") or 
        os.getenv("GOOGLE_API_KEY") or 
        os.getenv("PLANNER_API_KEY") or 
        os.getenv("WRITER_API_KEY") or 
        os.getenv("API")
    )
    MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-flash-latest")

    DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

    DEFAULT_LLM_MODEL = (
        _normalize_gemini_model(MODEL_NAME)
        if DEFAULT_LLM_PROVIDER == "google" else
        _normalize_non_google_model(os.getenv("LLM_MODEL", MODEL_NAME))
    )
    DEFAULT_LLM_API_KEY = GEMINI_API_KEY if DEFAULT_LLM_PROVIDER == "google" else (OPENROUTER_API_KEY or os.getenv("OPENAI_API_KEY"))

    # Agent-specific Configurations (defaults inherit from LLM_*)
    def _provider_value(var_name: str, default: str):
        return os.getenv(var_name, default).lower()

    def _model_value(env_var: str, default_model: str, provider: str):
        raw = os.getenv(env_var, default_model)
        if provider == "google":
            return _normalize_gemini_model(raw)
        return _normalize_non_google_model(raw)

    def _api_key_value(env_var: str, default_key: str, provider: str, openrouter_key: str):
        if provider == "google":
            return os.getenv(env_var, default_key) or default_key
        return os.getenv(env_var, openrouter_key) or openrouter_key or os.getenv("OPENAI_API_KEY")

    def _base_url(provider: str, openrouter_base_url: str):
        if provider == "google":
            return GEMINI_BASE_URL
        return openrouter_base_url

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "A GEMINI_API_KEY is required because the researcher agent is pinned to Gemini."
        )

    RESEARCHER_PROVIDER = "google"
    RESEARCHER_MODEL = _normalize_gemini_model(os.getenv("RESEARCHER_MODEL", DEFAULT_LLM_MODEL))
    RESEARCHER_API_KEY = os.getenv("RESEARCHER_API_KEY", GEMINI_API_KEY)
    RESEARCHER_BASE_URL = GEMINI_BASE_URL

    PLANNER_PROVIDER = _provider_value("PLANNER_PROVIDER", DEFAULT_LLM_PROVIDER)
    PLANNER_MODEL = _model_value("PLANNER_MODEL", DEFAULT_LLM_MODEL, PLANNER_PROVIDER)
    PLANNER_API_KEY = _api_key_value("PLANNER_API_KEY", DEFAULT_LLM_API_KEY, PLANNER_PROVIDER, OPENROUTER_API_KEY)
    PLANNER_BASE_URL = _base_url(PLANNER_PROVIDER, OPENROUTER_BASE_URL)

    WRITER_PROVIDER = _provider_value("WRITER_PROVIDER", DEFAULT_LLM_PROVIDER)
    WRITER_MODEL = _model_value("WRITER_MODEL", DEFAULT_LLM_MODEL, WRITER_PROVIDER)
    WRITER_API_KEY = _api_key_value("WRITER_API_KEY", DEFAULT_LLM_API_KEY, WRITER_PROVIDER, OPENROUTER_API_KEY)
    WRITER_BASE_URL = _base_url(WRITER_PROVIDER, OPENROUTER_BASE_URL)

    OPTIMIZER_PROVIDER = _provider_value("OPTIMIZER_PROVIDER", DEFAULT_LLM_PROVIDER)
    OPTIMIZER_MODEL = _model_value("OPTIMIZER_MODEL", DEFAULT_LLM_MODEL, OPTIMIZER_PROVIDER)
    OPTIMIZER_API_KEY = _api_key_value("OPTIMIZER_API_KEY", DEFAULT_LLM_API_KEY, OPTIMIZER_PROVIDER, OPENROUTER_API_KEY)
    OPTIMIZER_BASE_URL = _base_url(OPTIMIZER_PROVIDER, OPENROUTER_BASE_URL)

    QA_PROVIDER = _provider_value("QA_PROVIDER", DEFAULT_LLM_PROVIDER)
    QA_MODEL = _model_value("QA_MODEL", DEFAULT_LLM_MODEL, QA_PROVIDER)
    QA_API_KEY = _api_key_value("QA_API_KEY", DEFAULT_LLM_API_KEY, QA_PROVIDER, OPENROUTER_API_KEY)
    QA_BASE_URL = _base_url(QA_PROVIDER, OPENROUTER_BASE_URL)
