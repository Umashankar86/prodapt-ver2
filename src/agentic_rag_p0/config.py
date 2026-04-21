from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    structured_dir: Path
    doc_index_path: Path
    sqlite_db_path: Path
    llm_cache_path: Path
    llm_log_path: Path
    tavily_api_key: str | None
    vertex_project_id: str | None
    vertex_location: str
    gemini_fast_model: str
    gemini_pro_model: str
    continuity_enabled: bool
    llm_cache_enabled: bool = False


def load_settings() -> Settings:
    _load_dotenv(Path(".env"))
    return Settings(
        docs_dir=Path(os.environ.get("AGENTIC_RAG_DOCS_DIR", "data/docs")),
        structured_dir=Path(os.environ.get("AGENTIC_RAG_STRUCTURED_DIR", "data/structured")),
        doc_index_path=Path(os.environ.get("AGENTIC_RAG_DOC_INDEX", "artifacts/docs_index.json")),
        sqlite_db_path=Path(os.environ.get("AGENTIC_RAG_SQLITE_DB", "artifacts/structured.db")),
        llm_cache_path=Path(os.environ.get("AGENTIC_RAG_LLM_CACHE", "artifacts/llm_cache.json")),
        llm_log_path=Path(os.environ.get("AGENTIC_RAG_LLM_LOG", "artifacts/llm_responses.json")),
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        vertex_project_id=os.environ.get("VERTEX_PROJECT_ID"),
        vertex_location=os.environ.get("VERTEX_LOCATION", "us-central1"),
        gemini_fast_model=os.environ.get("GEMINI_FAST_MODEL", "gemini-2.5-flash"),
        gemini_pro_model=os.environ.get("GEMINI_PRO_MODEL", "gemini-2.5-pro"),
        continuity_enabled=os.environ.get("AGENTIC_RAG_CONTINUITY_ENABLED", "false").lower() == "true",
        llm_cache_enabled=os.environ.get("AGENTIC_RAG_ENABLE_LLM_CACHE", "false").lower() == "true",
    )
