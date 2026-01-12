import os
from typing import Optional

from agno.vectordb.qdrant import Qdrant
from agno.knowledge.embedder.google import GeminiEmbedder
from config.settings import Config


class _InMemoryKnowledge:
    """Very small in-memory fallback to keep the pipeline running without Qdrant."""

    def __init__(self):
        docs = []
        kb_path = os.path.join(os.path.dirname(__file__), "docs")
        for root, _, files in os.walk(kb_path):
            for file_name in files:
                if file_name.endswith((".md", ".txt")):
                    file_path = os.path.join(root, file_name)
                    with open(file_path, "r", encoding="utf-8") as handle:
                        docs.append(handle.read())
        self._documents = docs

    def search(self, query: str, limit: int = 3, **_):
        # The pipeline expects objects with `.content`; emulate that interface
        class _Doc:
            def __init__(self, text: str):
                self.content = text

        return [_Doc(text) for text in self._documents[:limit]]


_cached_vector_db: Optional[object] = None


def _use_in_memory_fallback(reason: str):
    print(f"[fallback] Falling back to in-memory knowledge base ({reason})")
    return _InMemoryKnowledge()


def get_knowledge_base():
    """
    Initializes and returns the vector DB instance.
    Falls back to an in-memory KB when Qdrant cannot be reached.
    """
    global _cached_vector_db
    if _cached_vector_db:
        return _cached_vector_db

    if not Config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY must be configured")

    if Config.QDRANT_URL == ":memory:":
        _cached_vector_db = _use_in_memory_fallback("QDRANT_URL=:memory:")
        return _cached_vector_db

    try:
        _cached_vector_db = Qdrant(
            collection=Config.COLLECTION_NAME,
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            embedder=GeminiEmbedder(api_key=Config.GEMINI_API_KEY),
        )
        # Trigger a lightweight call to surface connection issues early
        if hasattr(_cached_vector_db, "client"):
            _cached_vector_db.client.get_collections()
        return _cached_vector_db
    except Exception as exc:
        _cached_vector_db = _use_in_memory_fallback(str(exc))
        return _cached_vector_db
