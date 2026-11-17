# CandiMate

> A recruiter-facing RAG chatbot that answers questions about your resume and
> projects using grounded, source-aware responses.

Built with **Python**, **Streamlit**, **LangChain**, **Chroma**, and either the
**Gemini Developer API** or **OpenAI API**.

## Why this project

A resume is static. This application turns your resume and project write-ups into
an interactive portfolio that recruiters can query directly:

- “What was your contribution to this project?”
- “Why did you choose this architecture?”
- “How did you measure the result?”
- “What would you improve next?”

The assistant retrieves relevant portfolio content before generating an answer and
shows the source files used for that answer.

## Features

- Resume ingestion from **PDF**, Markdown, or plain text
- Project write-up ingestion from Markdown, PDF, or plain text
- Gemini and OpenAI provider support
- Persistent Chroma vector retrieval
- Automatic index refresh when portfolio files change
- Conversation history and follow-up questions
- Source labels for every answer
- Demo mode without an API key
- Docker and Streamlit Community Cloud deployment
- Automated tests through GitHub Actions

## Project structure

```text
.
├── app.py                         # CandiMate Streamlit application
├── data/                          # Portfolio knowledge files
│   ├── projects.md
├── portfolio_chatbot/
│   ├── chat.py                    # Grounded answer generation
│   ├── config.py                  # Environment configuration
│   └── knowledge.py               # PDF loading, chunking, and retrieval
├── tests/                         # Focused automated tests
├── .env.example                   # Safe configuration template
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

Generated files such as `.venv/`, `.chroma/`, caches, and `.env` are intentionally
excluded from GitHub by `.gitignore`.

## Quick start on Windows

Run these commands from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Configure Gemini

Create a Gemini Developer API key in
[Google AI Studio](https://aistudio.google.com/apikey), then set these values in
`.env`:

```env
AI_PROVIDER=gemini
GOOGLE_API_KEY=your-google-api-key
GEMINI_CHAT_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

The Gemini web subscription and Gemini Developer API access may have separate
quotas and billing. Confirm that the selected model is available for your API
project.

## Configure OpenAI

```env
AI_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Only the selected provider's key is required.

## Customize the knowledge base

1. Add your resume and project write-ups to `data/`.
2. Use `.pdf`, `.md`, or `.txt` files.
3. Add as many knowledge files as needed; filenames do not need to follow a
   specific convention.
4. Include your role, technical decisions, trade-offs, links, and measurable outcomes.
5. Restart the app after editing.

PDFs must contain selectable text. Scanned, image-only PDFs require OCR before
they can be indexed.

## Tests

```powershell
pytest
```

The test suite covers document loading, PDF handling, chunking, fallback retrieval,
and provider response formatting.

## Docker

```powershell
docker build -t candimate .
docker run --rm -p 8501:8501 --env-file .env candimate
```

## Deploy to Streamlit Community Cloud

1. Push this project folder to a GitHub repository.
2. Create a new Streamlit Community Cloud app.
3. Select the repository and set the entry point to `app.py`.
4. Add the matching provider key under **App settings → Secrets**.

Example Gemini secret:

```toml
AI_PROVIDER = "gemini"
GOOGLE_API_KEY = "your-google-api-key"
GEMINI_CHAT_MODEL = "gemini-3.8-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
```

Never commit `.env` or API keys. Rotate a key immediately if it is exposed.

## GitHub checklist

- Replace all personal placeholder content before publishing.
- Review generated answers against your source documents.
- Add your portfolio URL and contact links to the Streamlit UI.
- Confirm the repository contains no `.env`, API keys, `.venv`, or `.chroma` files.
- Enable the included GitHub Actions workflow and confirm tests pass.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
"# CandiMate" 
