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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    print("Error: Missing API Keys in .env")
    exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Indexes (V4)
TEXT_INDEX_NAME = "website-text-v4"
IMAGE_INDEX_NAME = "website-images-v4"

text_idx = pc.Index(TEXT_INDEX_NAME)
image_idx = pc.Index(IMAGE_INDEX_NAME)

# 2. Setup Embedding Model
print("Loading BAAI/bge-base-en-v1.5 on CPU...")
device = "cpu" # Stable for ingestion
model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)

def get_bge_embedding(text):
    instruction = "represent the document for retrieval: "
    return model.encode(instruction + text).tolist()

def generate_strict_caption(page_text, page_num, img_idx, project_name):
    """Use Llama 3.3 70B to generate a high-quality, specific, non-generic caption."""
    prompt = f"""You are a technical document analyst. You are looking at an image in a document titled "{project_name}".
The text around the image is:
---
{page_text[:3000]}
---
TASK: Provide a SPECIFIC, TECHNICAL caption for this image (max 25 words).
- DO NOT use words like "photograph", "image", "example", or "figure".
- Identify EXACTLY what is shown: a specific textile motif (name it), a tool (name it), a graph (what data?), or a process (what stage?).
- Use terms from the text.
- If the text mentions "Figure X" or "Table X", use that description but make it natural.
- NEVER use generic placeholders like "Textile photograph related to...".
- If the text contains data (percentages, kg CO2, etc.) that clearly belongs to this image/chart, include it.

Caption:"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise technical analyzer. You never use generic filler text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=80,
        )
        caption = response.choices[0].message.content.strip().strip('"')
        # Final safety check against generic patterns
        if "photograph related to" in caption.lower() or "image of" in caption.lower()[:8]:
            # Try once more with even stricter instruction
            prompt += "\n\nCRITICAL: YOUR PREVIOUS ATTEMPT WAS TOO GENERIC. BE MORE SPECIFIC ABOUT THE TEXTILE OR DATA."
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
            )
            caption = response.choices[0].message.content.strip().strip('"')
        return caption
    except Exception as e:
        print(f"  Error generating caption: {e}")
        return f"Technical detail from {project_name}, page {page_num}"

def process_pdf(pdf_filename):
    pdf_path = os.path.join(BASE_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_filename} not found.")
        return

    project_name = pdf_filename.replace(".pdf", "")
    # Generate a clean ID slug
    pdf_id_slug = "".join([c for c in project_name if c.isalnum()])[:20]
    
    # Classification logic
    doc_type = "supplementary"
    if "carbon" in project_name.lower():
        doc_type = "supplementary" # Carbon footprint is cross-cutting
    elif "baluchari" in project_name.lower() or "saree" in project_name.lower():
        doc_type = "project_report"
    
    print(f"\n=== Processing: {project_name} | Slug: {pdf_id_slug} | Type: {doc_type} ===")
    
    doc = fitz.open(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    static_images_dir = os.path.join(BASE_DIR, "static", "images")
    os.makedirs(static_images_dir, exist_ok=True)
    
    all_text_chunks = []
    all_image_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        
        # Text Processing
        if page_text.strip():
            chunks = text_splitter.split_text(page_text)
            for i, chunk in enumerate(chunks):
                all_text_chunks.append({
                    "id": f"v4_{pdf_id_slug}_pg{page_num+1}_ch{i}",
                    "text": chunk,
                    "metadata": {
                        "project": project_name,
                        "page": page_num + 1,
                        "text": chunk,
                        "doc_type": doc_type
                    }
                })
        
        # Image Processing
        images = page.get_images(full=True)
        for img_idx, img in enumerate(images):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                
                # Save image locally
                img_filename = f"{pdf_id_slug}_pg{page_num+1}_img{img_idx}.{ext}"
                img_path = os.path.join(static_images_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                    
                # Generate strict caption
                print(f"  Generating caption for Page {page_num+1}, Image {img_idx}...")
                caption = generate_strict_caption(page_text, page_num + 1, img_idx, project_name)
                print(f"    -> {caption}")
                
                all_image_data.append({
                    "id": f"v4_img_{pdf_id_slug}_pg{page_num+1}_{img_idx}",
                    "caption": caption,
                    "metadata": {
                        "project": project_name,
                        "page": page_num + 1,
                        "image_url": f"/static/images/{img_filename}",
                        "description": caption,
                        "doc_type": doc_type
                    }
                })
                time.sleep(0.5) # Rate limiting
            except Exception as e:
                print(f"  Skipping image {img_idx} on page {page_num+1}: {e}")

    # Upload Text
    if all_text_chunks:
        print(f"  Uploading {len(all_text_chunks)} text chunks...")
        for i in range(0, len(all_text_chunks), 20):
            batch = all_text_chunks[i:i+20]
            to_upsert = []
            for item in batch:
                vec = get_bge_embedding(item["text"])
                to_upsert.append((item["id"], vec, item["metadata"]))
            text_idx.upsert(vectors=to_upsert)
            
    # Upload Images
    if all_image_data:
        print(f"  Uploading {len(all_image_data)} images...")
        for i in range(0, len(all_image_data), 20):
            batch = all_image_data[i:i+20]
            to_upsert = []
            for item in batch:
                vec = get_bge_embedding(item["caption"])
                to_upsert.append((item["id"], vec, item["metadata"]))
            image_idx.upsert(vectors=to_upsert)

    print(f"SUCCESS: Ingested {pdf_filename}")

if __name__ == "__main__":
    new_pdfs = [
        "Carbon footprint of Heritage products.pdf",
        "Data for AI chatbot.pdf"
    ]
    for pdf in new_pdfs:
        process_pdf(pdf)
