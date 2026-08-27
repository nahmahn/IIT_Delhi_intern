import os
import json
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

print("1. Loading local embedding model (nomic-embed-text)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("2. Connecting to Pinecone...")
vectorstore = PineconeVectorStore(
    index_name="ask-textile",
    embedding=embeddings
)

print("3. Loading Scraped NPTEL HTML Lectures...")
# The newly scraped file from the data folder
dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "textile_html_lectures.json")

if not os.path.exists(dataset_path):
    print(f"Error: Could not find {dataset_path}.")
    print("Wait for the scraper to finish running!")
    exit(1)

with open(dataset_path, "r", encoding="utf-8") as f:
    lectures = json.load(f)

print(f"Loaded {len(lectures)} raw lectures. Splitting into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    length_function=len,
)

docs = []
ids = []
for lec in lectures:
    text = lec.get("content", "")
    if not text: continue
    
    # Chunk the massive text into smaller vectors
    chunks = splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        metadata = {
            "course_id": lec.get("course_id", ""),
            "course_title": lec.get("course_title", ""),
            "professor": lec.get("professor", ""),
            "institute": lec.get("institute", ""),
            "lecture_name": lec.get("lecture_name", ""),
            "chunk_index": i
        }
        docs.append(Document(
            page_content=chunk,
            metadata=metadata
        ))
        # Stable unique ID based on course, lecture, and chunk index
        chunk_id = f"{metadata['course_id']}_{metadata['lecture_name']}_{i}".replace(" ", "_")
        ids.append(chunk_id)

print(f"Generated {len(docs)} contextual chunks! Resuming upload to Pinecone...")

batch_size = 100
for i in range(0, len(docs), batch_size):
    batch_docs = docs[i : i + batch_size]
    batch_ids = ids[i : i + batch_size]
    print(f"  -> Processing batch {i} to {i+len(batch_docs)} out of {len(docs)}")
    vectorstore.add_documents(documents=batch_docs, ids=batch_ids)

print("Vector Database Migration Fully Complete (Resumed & Sync'd)!")
