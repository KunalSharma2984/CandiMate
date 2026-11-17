# Projects

## CandiMate

### Problem

Recruiters often have questions about a candidate's projects that a static resume
cannot answer quickly.

### Approach

This application loads PDF, Markdown, and text portfolio notes, preserves their section metadata,
splits them into retrieval-sized chunks, and uses OpenAI embeddings with a Chroma
vector store. A LangChain chat model receives only the retrieved CandiMate context,
which reduces unsupported claims. A keyword-based preview mode keeps the app useful
without exposing an API key during local setup.

### Stack

Python, Streamlit, LangChain, OpenAI API, Chroma, pytest, Docker.

### Outcome

The app provides a shareable recruiter demo with cited sources and a replaceable
knowledge base. Replace this outcome with your real usage metrics, latency, or
deployment URL.

## Add Your Projects

Create additional Markdown files in `data/`. Use headings for project names and
include the problem, role, technical decisions, trade-offs, results, and links.
