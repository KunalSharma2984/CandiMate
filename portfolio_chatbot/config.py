"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / ".chroma"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the chatbot."""

    provider: str
    openai_api_key: str | None
    google_api_key: str | None
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    data_dir: Path = DATA_DIR
    chroma_dir: Path = CHROMA_DIR


def get_settings() -> Settings:
    """Read settings, applying safe defaults for local development."""
    provider = os.getenv("AI_PROVIDER", "openai").lower()
    chat_model = os.getenv(
        "GEMINI_CHAT_MODEL" if provider == "gemini" else "OPENAI_CHAT_MODEL",
        "gemini-3.8-flash" if provider == "gemini" else "gpt-4o-mini",
    )
    if provider == "gemini" and chat_model in {
        "gemini-2.5-flash",
        "models/gemini-2.5-flash",
    }:
        chat_model = "gemini-3.8-flash"

    return Settings(
        provider=provider,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        google_api_key=os.getenv("GOOGLE_API_KEY") or None,
        chat_model=chat_model,
        embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL" if provider == "gemini"
            else "OPENAI_EMBEDDING_MODEL",
            "models/gemini-embedding-001" if provider == "gemini" else "text-embedding-3-small",
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "5")),
    )
