"""Document loading, chunking, and vector retrieval."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import Settings

SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}


def load_documents(data_dir: Path) -> list[Document]:
    """Load portfolio files in a deterministic order with source metadata."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {data_dir}")

    files = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No .md, .pdf, or .txt files found in {data_dir}")

    documents: list[Document] = []
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    for path in files:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        else:
            text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        if path.suffix.lower() == ".md":
            parts = markdown_splitter.split_text(text)
        else:
            parts = [Document(page_content=text)]
        for part in parts:
            part.metadata.update({"source": path.name, "path": str(path.relative_to(data_dir))})
            documents.append(part)
    return documents


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    """Split loaded documents while retaining source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def knowledge_base_fingerprint(settings: Settings) -> str:
    """Return a content hash used to invalidate Streamlit's cached retriever."""
    digest = hashlib.sha256()
    for path in sorted(
        p for p in settings.data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ):
        digest.update(str(path.relative_to(settings.data_dir)).encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(
        f"{settings.provider}:{settings.embedding_model}:{settings.chunk_size}:"
        f"{settings.chunk_overlap}".encode()
    )
    return digest.hexdigest()


def build_retriever(settings: Settings, fingerprint: str | None = None):
    """Build a Chroma retriever from the current portfolio documents."""
    if settings.provider not in {"openai", "gemini"}:
        raise ValueError("AI_PROVIDER must be either 'openai' or 'gemini'")
    has_api_key = (
        settings.openai_api_key if settings.provider == "openai" else settings.google_api_key
    )
    if not has_api_key:
        return KeywordRetriever(split_documents(load_documents(settings.data_dir), settings), settings.retrieval_k)

    from langchain_chroma import Chroma

    documents = split_documents(load_documents(settings.data_dir), settings)
    if settings.provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
    else:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_name=f"portfolio-{(fingerprint or 'default')[:24]}",
    )
    return vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})


class KeywordRetriever:
    """Small dependency-free fallback for previews without an OpenAI key."""

    def __init__(self, documents: list[Document], k: int) -> None:
        self.documents = documents
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        terms = {term.lower() for term in query.split() if len(term) > 2}
        ranked = sorted(
            self.documents,
            key=lambda doc: sum(term in doc.page_content.lower() for term in terms),
            reverse=True,
        )
        return ranked[: self.k]
