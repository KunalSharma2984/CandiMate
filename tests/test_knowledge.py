from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader, PdfWriter

from portfolio_chatbot.config import Settings
from portfolio_chatbot.chat import _content_to_text
from portfolio_chatbot.knowledge import KeywordRetriever, load_documents, split_documents


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        provider="openai",
        openai_api_key=None,
        google_api_key=None,
        chat_model="test",
        embedding_model="test",
        chunk_size=80,
        chunk_overlap=10,
        retrieval_k=2,
        data_dir=tmp_path,
        chroma_dir=tmp_path / ".chroma",
    )


def test_load_documents_preserves_sources(tmp_path: Path) -> None:
    (tmp_path / "resume.md").write_text("# Experience\n\nBuilt APIs.", encoding="utf-8")
    documents = load_documents(tmp_path)
    assert documents
    assert documents[0].metadata["source"] == "resume.md"
    assert "Built APIs" in documents[0].page_content


def test_load_documents_extracts_pdf_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "resume.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as file:
        writer.write(file)

    documents = load_documents(tmp_path)

    assert PdfReader(str(pdf_path)).pages
    assert documents == []


def test_split_documents_respects_small_chunks(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    documents = split_documents([Document(page_content="one " * 100)], settings)
    assert len(documents) > 1
    assert all(len(document.page_content) <= settings.chunk_size for document in documents)


def test_keyword_retriever_ranks_matching_content() -> None:
    retriever = KeywordRetriever(
        [
            Document(page_content="Python APIs", metadata={"source": "a.md"}),
            Document(page_content="Cooking recipes", metadata={"source": "b.md"}),
        ],
        k=1,
    )
    result = retriever.invoke("Python")
    assert result[0].metadata["source"] == "a.md"


def test_content_to_text_hides_provider_metadata() -> None:
    content = [
        {"type": "text", "text": "Hello! I am the portfolio assistant.", "extras": {"signature": "secret"}},
    ]
    assert _content_to_text(content) == "Hello! I am the portfolio assistant."
