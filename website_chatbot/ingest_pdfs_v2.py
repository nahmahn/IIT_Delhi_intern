import fitz  # PyMuPDF
import os
import io
import time
import requests
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch

# 1. Setup Environment
# Look for .env in website_chatbot or parent directory
BASE_DIR = os.path.dirname(__file__)
env_path = os.path.join(BASE_DIR, ".env")
if not os.path.exists(env_path):
    env_path = os.path.join(BASE_DIR, "..", "ask_textile", ".env")
load_dotenv(env_path)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    print("Error: Missing PINECONE_API_KEY")
    exit(1)

# 2. Setup Pinecone Indexes
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define our V2 indexes
TEXT_INDEX = "website-text-v2"

existing_indexes = pc.list_indexes().names()

if TEXT_INDEX not in existing_indexes:
    print(f"Index {TEXT_INDEX} missing. Creating it now...")
    pc.create_index(
        name=TEXT_INDEX,
        dimension=768, # BGE-base dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

def wait_index(name):
    print(f"Waiting for index {name} to be ready...")
    while not pc.describe_index(name).status["ready"]:
        time.sleep(1)
    print(f"Index {name} is ready.")

wait_index(TEXT_INDEX)
text_idx = pc.Index(TEXT_INDEX)

# 3. Setup Embedding Models
print("Loading Local HD Text Embeddings (BAAI/bge-base-en-v1.5)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
text_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)

# 4. Extract Data from PDFs
BASE_DIR = os.path.dirname(__file__)
pdf_files = [f for f in os.listdir(BASE_DIR) if f.endswith(".pdf")]

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

all_text_chunks = []   

for pdf_idx, pdf_file in enumerate(pdf_files):
    pdf_path = os.path.join(BASE_DIR, pdf_file)
    project_name = pdf_file.replace(".pdf", "")
    print(f"Processing PDF: {project_name}")
    
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract Text
        content = page.get_text()
        if content.strip():
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                all_text_chunks.append({
                    "id": f"doc_{pdf_idx}_pg_{page_num}_ch_{i}_v2",
                    "text": chunk,
                    "metadata": {
                        "project": project_name,
                        "page": page_num + 1,
                        "text": chunk 
                    }
                })

# 5. Upload to Pinecone
print(f"Found {len(all_text_chunks)} text chunks. Uploading to {TEXT_INDEX}...")
batch_size = 32

for i in range(0, len(all_text_chunks), batch_size):
    batch = all_text_chunks[i:i+batch_size]
    texts = [item["text"] for item in batch]
    instruction = "represent the document for retrieval: "
    vectors = text_model.encode([instruction + t for t in texts]).tolist()
    
    to_upsert = []
    for j in range(len(batch)):
        to_upsert.append((batch[j]["id"], vectors[j], batch[j]["metadata"]))
        
    text_idx.upsert(vectors=to_upsert)
    print(f"  -> Uploaded text batch {i} to {i+len(batch)}")

print(f"\nSUCCESS: PDFs ingested into {TEXT_INDEX}")
print(f"Image retrieval will continue using existing 'website-images-text' index.")

print(f"\nSUCCESS: PDFs ingested into {TEXT_INDEX}")
print(f"You can now point your RAG Reranker to this new high-precision data.")
