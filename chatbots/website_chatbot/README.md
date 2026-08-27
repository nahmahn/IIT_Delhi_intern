# Website chatbot

FastAPI RAG chatbot for the Textile Dept website. It answers questions about the department's heritage textile documentation — the source PDFs (Baluchari, Muslin, Negamam, Phulkari, carbon footprint, heritage product locations, and SHRI centre info) are included in this folder. Supports multilingual responses via the `language` field of the chat request.

## How it works

- `ingest_v4_new.py` — one-time ingestion: parses the PDFs, chunks them, embeds the chunks, and indexes them in **Pinecone**
- `rag.py` — query pipeline: retrieves relevant chunks from Pinecone and generates an answer with **Groq**, optionally returning related images
- `app.py` — FastAPI app: serves the static frontend (`index.html` at `/`), the `/chat` endpoint, and images under `/static/images`
- `Dockerfile` — container build used by the Hugging Face Space (port 7860)

## Running locally

```bash
pip install -r requirements.txt
# create .env in this folder with:
#   PINECONE_API_KEY=...
#   GROQ_API_KEY=...
python ingest_v4_new.py          # one-time: index the PDFs
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Running with Docker

```bash
docker build -t textile-chatbot .
docker run -p 7860:7860 --env-file .env textile-chatbot
```

## Deployment

Deployed as a Docker-based Hugging Face Space. The Space configuration lives in the YAML front-matter of the repository root `README.md` (`app_dir: chatbots/website_chatbot`, `app_port: 7860`).
