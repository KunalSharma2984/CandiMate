"""Streamlit entry point for CandiMate."""

from __future__ import annotations

import streamlit as st

from portfolio_chatbot.chat import answer_question
from portfolio_chatbot.config import get_settings
from portfolio_chatbot.knowledge import build_retriever, knowledge_base_fingerprint

st.set_page_config(
    page_title="CandiMate",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings = get_settings()


@st.cache_resource(show_spinner="Indexing portfolio knowledge…")
def get_retriever(fingerprint: str):
    return build_retriever(settings, fingerprint)


def reset_chat() -> None:
    st.session_state.messages = []


st.title("💼 CandiMate")
st.caption("Your recruiter-friendly AI assistant, grounded in your portfolio knowledge files.")

with st.sidebar:
    st.header("About CandiMate")
    st.write(
        "Ask about experience, projects, technical decisions, impact, or the skills "
        "behind the work. Answers include the portfolio sources used."
    )
    configured_key = (
        settings.openai_api_key if settings.provider == "openai" else settings.google_api_key
    )
    if configured_key:
        st.success(f"{settings.provider.title()} is configured")
    else:
        st.warning(f"Demo mode: add the {settings.provider.upper()} API key for AI answers")
    st.button("Clear conversation", on_click=reset_chat, use_container_width=True)
    st.divider()
    st.markdown("**Customize it**")
    st.markdown(
        "Add your resume and project notes to `data/`, then refresh the app."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

question = st.chat_input("Ask a question about my background…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            retriever = get_retriever(knowledge_base_fingerprint(settings))
            response = answer_question(question, st.session_state.messages, retriever, settings)
            st.markdown(response.answer)
            if response.sources:
                with st.expander("Sources"):
                    for source in response.sources:
                        st.markdown(f"- `{source}`")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                }
            )
        except Exception as exc:
            st.error("The assistant could not answer right now. Check the terminal logs and configuration.")
            st.exception(exc)
