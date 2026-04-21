import fitz  # PyMuPDF
import os
import io
import time
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
import torch

# 1. Setup Environment
BASE_DIR = os.path.dirname(__file__)
env_path = os.path.join(BASE_DIR, ".env")
if not os.path.exists(env_path):
    env_path = os.path.join(BASE_DIR, "..", "ask_textile", ".env")
load_dotenv(env_path)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    print("Error: Missing API Keys in .env")
    exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Indexes
TEXT_INDEX_NAME = "website-text-v4"
IMAGE_INDEX_NAME = "website-images-v4"

text_idx = pc.Index(TEXT_INDEX_NAME)
image_idx = pc.Index(IMAGE_INDEX_NAME)

# --- CLASSIFICATION LOGIC ---
DOC_TYPE_RULES = {
    "project_report": ["baluchari", "muslin", "negamam", "phulkari", "maheshwari"],
    "dept_info": ["shri", "centre", "department"],
}

def classify_doc_type(project_name):
    name_lower = project_name.lower()
    for doc_type, keywords in DOC_TYPE_RULES.items():
        if any(kw in name_lower for kw in keywords):
            return doc_type
    return "supplementary"

# 2. Setup Embedding Model
print("Loading BAAI/bge-base-en-v1.5...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)

def get_bge_embedding(text):
    instruction = "represent the document for retrieval: "
    return model.encode(instruction + text).tolist()

import sys

# 3. Target PDF
if len(sys.argv) > 1:
    PDF_FILENAME = sys.argv[1]
else:
    PDF_FILENAME = "Carbon footprint of Heritage products.pdf"
PDF_PATH = os.path.join(BASE_DIR, PDF_FILENAME)
PROJECT_NAME = PDF_FILENAME.replace(".pdf", "")
DOC_TYPE = classify_doc_type(PROJECT_NAME)
# Assigning index 6 to avoid collision with existing 0-5
PDF_ID_IDX = 6

if not os.path.exists(PDF_PATH):
    print(f"Error: {PDF_FILENAME} not found in {BASE_DIR}")
    exit(1)

# 4. Processing
print(f"=== Processing: {PROJECT_NAME} | Type: {DOC_TYPE} ===")
doc = fitz.open(PDF_PATH)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

STATIC_IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)

all_text_chunks = []
all_image_data = []

def generate_llama_caption(page_text, page_num, img_idx):
    """Use Llama 3.3 70B to generate a high-quality, data-aware caption."""
    prompt = f"""You are describing an image extracted from a document about "{PROJECT_NAME}".
Based ONLY on the surrounding text below, write a brief, natural caption for what this image likely depicts.

RULES:
1. Keep it under 20 words.
2. Always mention the specific textile type (e.g., "Negamam saree", "Phulkari embroidery").
3. ONLY describe it as a chart/graph if the text explicitly says "Figure X:" or "Table X:" with specific data labels right next to this image position.
4. Otherwise, assume it is a photograph and describe the visual subject.
5. If unsure, write: "Textile photograph related to {PROJECT_NAME}."

Page text snippet:
{page_text[:2500]}

Caption:"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for more factual captions
            max_tokens=60,
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print(f"  Error generating caption: {e}")
        return f"Figure from {PROJECT_NAME}, page {page_num}"

# Iterate through pages
for page_num in range(len(doc)):
    page = doc[page_num]
    page_text = page.get_text()
    
    # Text Processing
    if page_text.strip():
        chunks = text_splitter.split_text(page_text)
        for i, chunk in enumerate(chunks):
            all_text_chunks.append({
                "id": f"doc_{PDF_ID_IDX}_pg_{page_num}_ch_{i}_v2",
                "text": chunk,
                "metadata": {
                    "project": PROJECT_NAME,
                    "page": page_num + 1,
                    "text": chunk,
                    "doc_type": DOC_TYPE
                }
            })
    
    # Image Processing
    images = page.get_images(full=True)
    for img_idx, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        ext = base_image["ext"]
        
        # Save image locally
        img_filename = f"{PROJECT_NAME}_pg{page_num+1}_img{img_idx}.{ext}".replace(" ", "_")
        img_path = os.path.join(STATIC_IMAGES_DIR, img_filename)
        with open(img_path, "wb") as f:
            f.write(image_bytes)
            
        # Generate high-quality caption with Llama 70B
        print(f"  Generating caption for Page {page_num+1}, Image {img_idx}...")
        caption = generate_llama_caption(page_text, page_num + 1, img_idx)
        print(f"    -> {caption}")
        
        all_image_data.append({
            "id": f"img_{PDF_ID_IDX}_{page_num}_{img_idx}",
            "caption": caption,
            "metadata": {
                "project": PROJECT_NAME,
                "page": page_num + 1,
                "image_url": f"/static/images/{img_filename}",
                "description": caption,
                "doc_type": DOC_TYPE
            }
        })
        time.sleep(0.5) # Rate limiting for Groq

# 5. Final Ingestion
print(f"\n--- Uploading {len(all_text_chunks)} text chunks to {TEXT_INDEX_NAME} ---")
for i in range(0, len(all_text_chunks), 20):
    batch = all_text_chunks[i:i+20]
    to_upsert = []
    for item in batch:
        vec = get_bge_embedding(item["text"])
        to_upsert.append((item["id"], vec, item["metadata"]))
    text_idx.upsert(vectors=to_upsert)
    print(f"  Uploaded text batch {i}-{i+len(batch)}")

print(f"\n--- Uploading {len(all_image_data)} images to {IMAGE_INDEX_NAME} ---")
for i in range(0, len(all_image_data), 20):
    batch = all_image_data[i:i+20]
    to_upsert = []
    for item in batch:
        # Embed the CAPTION, not the image! (This matches the website-images-text system)
        vec = get_bge_embedding(item["caption"])
        to_upsert.append((item["id"], vec, item["metadata"]))
    image_idx.upsert(vectors=to_upsert)
    print(f"  Uploaded image batch {i}-{i+len(batch)}")

print(f"\nSUCCESS: Ingested {PROJECT_NAME} into Pinecone with Llama-70B captions.")
