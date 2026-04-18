from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from rag import process_query

app = FastAPI(title="Textile Dept RAG API", description="Project Chatbot Backend")

# Setup CORS (so frontend can call it from another port or domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static images directory to serve extracted images
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(STATIC_DIR, exist_ok=True) # Ensure it exists if empty

app.mount("/static/images", StaticFiles(directory=STATIC_DIR), name="images")

class ChatRequest(BaseModel):
    query: str
    language: str = "English"  # Option for Hindi, Tamil, etc.

@app.get("/")
def read_root():
    return {"message": "Textile Dept RAG API is running!"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    print(f"\n[API] Received query: '{req.query}' in {req.language}")
    
    # Process through RAG
    result = process_query(req.query, req.language)
    return result

# You can run this file via: uvicorn app:app --reload --port 8000
