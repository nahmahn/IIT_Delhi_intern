import os
import fitz
import json
import torch
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
import io

# --- CONFIGURATION & SETUP ---
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(STATIC_DIR, exist_ok=True)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Use CPU for ingestion to avoid local DLL issues
DEVICE = "cpu"
print(f"Ingest: Loading Embedding Model (BGE-base) on {DEVICE}...")
text_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=DEVICE)

# --- INDEX CREATION LOGIC ---
TEXT_INDEX_NAME = "website-text-v4"
IMAGE_INDEX_NAME = "website-images-v4"

def ensure_index(name, dim):
    if name not in [idx.name for idx in pc.list_indexes()]:
        print(f"Ingest: Creating index '{name}'...")
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(name).status['ready']:
            time.sleep(1)
    return pc.Index(name)

text_idx = ensure_index(TEXT_INDEX_NAME, 768)
image_idx = ensure_index(IMAGE_INDEX_NAME, 768) # We'll use text-based image search

# --- CLASSIFICATION LOGIC (Mentor's Guidance) ---
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

# --- SMART CAPTIONING (Chart detection) ---
def generate_smart_caption(page_text, page_num, img_idx, project_name):
    """Use Llama-70B to generate a data-aware caption."""
    prompt = f"""You are describing an image extracted from a document about "{project_name}".
Based ONLY on the surrounding text below, write a brief, natural caption for what this image likely depicts.

RULES:
1. Keep it under 20 words.
2. Always mention the specific textile type (e.g., "Negamam saree", "Phulkari embroidery").
3. ONLY describe it as a chart/graph or a piechart if it looks like one or also if the text explicitly says "Figure X:" or "Table X:" with specific data labels right next to this image position.
4. Otherwise, assume it is a photograph and describe the visual subject.
5. If unsure, write: "Textile photograph related to {project_name}."

Page text snippet:
{page_text[:2500]}

Caption:"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You are a precise technical document analyzer."},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=60,
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print(f"  Error generating caption: {e}")
        return f"Image from {project_name}, page {page_num}"

# --- MAIN INGESTION LOOP ---
def process_pdf(pdf_path):
    pdf_filename = os.path.basename(pdf_path)
    project_name = pdf_filename.replace(".pdf", "")
    pdf_id = "".join([c for c in project_name if c.isalnum()])[:20]
    doc_type = classify_doc_type(project_name)
    
    print(f"\n=== Processing: {project_name} | Type: {doc_type} ===")
    
    doc = fitz.open(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    text_chunks = []
    image_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        
        # 1. Process Text
        chunks = splitter.split_text(page_text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"v3_{pdf_id}_pg{page_num+1}_ch{i}"
            vec = text_model.encode(chunk).tolist()
            text_chunks.append({
                "id": chunk_id,
                "values": vec,
                "metadata": {
                    "project": project_name,
                    "page": page_num + 1,
                    "text": chunk,
                    "doc_type": doc_type
                }
            })

        # 2. Process Images
        images = page.get_images(full=True)
        for img_idx, img_info in enumerate(images):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            
            # Save Image
            img_filename = f"{pdf_id}_pg{page_num+1}_img{img_idx}.{ext}"
            img_path = os.path.join(STATIC_DIR, img_filename)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            
            # Generate Caption
            print(f"  Capturing Image {img_idx} on Page {page_num+1}...")
            caption = generate_smart_caption(page_text, page_num + 1, img_idx, project_name)
            print(f"    -> {caption}")
            
            # Embed and store
            img_id = f"v3_img_{pdf_id}_pg{page_num+1}_{img_idx}"
            img_vec = text_model.encode(caption).tolist()
            image_data.append({
                "id": img_id,
                "values": img_vec,
                "metadata": {
                    "project": project_name,
                    "page": page_num + 1,
                    "image_url": f"/static/images/{img_filename}",
                    "description": caption,
                    "doc_type": doc_type
                }
            })

    # Batch Upsert
    if text_chunks:
        print(f"  Upserting {len(text_chunks)} text chunks...")
        text_idx.upsert(vectors=text_chunks)
    if image_data:
        print(f"  Upserting {len(image_data)} images...")
        image_idx.upsert(vectors=image_data)

if __name__ == "__main__":
    pdfs = [f for f in os.listdir(BASE_DIR) if f.endswith(".pdf")]
    print(f"Starting V3 Re-ingestion for {len(pdfs)} PDFs...")
    for pdf in pdfs:
        process_pdf(os.path.join(BASE_DIR, pdf))
    print("\nSUCCESS: All data migrated to V3 with metadata classification.")
