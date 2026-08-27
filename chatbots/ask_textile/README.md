# Ask Textile

Three-tier RAG platform for asking questions about textile course content (`data/` and `RAG/textile_courses.json`, including a YouTube-augmented variant).

## Architecture

```
frontend (React)  ->  middleware (Node/Prisma)  ->  RAG service (Python + Pinecone + Groq)
```

An architecture diagram is in [RAG/architecture.svg](RAG/architecture.svg).

| Folder | What it is |
|---|---|
| `frontend/` | React + TypeScript + Vite + Tailwind chat UI |
| `middleware/` | Node/TypeScript API layer with Prisma for persistence — see [middleware/SETUP.md](middleware/SETUP.md) for setup |
| `RAG/` | Python retrieval service: chunking (`chunk.py`), Pinecone ingestion (`ingest_pinecone.py`), retriever (`retreiver.py`), LLM prompts (`prompts.py`, `llm.py`), entry point (`main.py`) |
| `data/` | Course content used for ingestion |

## Evaluation

The RAG pipeline is evaluated with **RAGAS** (`RAG/ragas_evaluation.py`); the latest scores are in `RAG/ragas_results.csv`.

## Setup

1. Create a `.env` in this folder with `PINECONE_API_KEY` and `GROQ_API_KEY` (gitignored, never committed).
2. RAG service: `pip install -r requirements.txt`, then run `python RAG/ingest_pinecone.py` once to index the course content.
3. Middleware: follow [middleware/SETUP.md](middleware/SETUP.md).
4. Frontend: `cd frontend && npm install && npm run dev`.
