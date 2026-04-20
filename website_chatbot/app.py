from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from rag import process_query

app = FastAPI(title="Textile Dept RAG API", description="Project Chatbot Backend")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount the static images directory
app.mount("/static/images", StaticFiles(directory=STATIC_DIR), name="images")

class ChatRequest(BaseModel):
    query: str
    language: str = "English"

@app.get("/")
def read_root():
    """Serve the frontend index.html from the root."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    print(f"\n[API] Received query: '{req.query}' in {req.language}")
    result = process_query(req.query, req.language)
    return result

# Run via: uvicorn app:app --host 0.0.0.0 --port 8000
