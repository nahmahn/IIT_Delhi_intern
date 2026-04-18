import fitz  # PyMuPDF
import os
import io
import time
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import CLIPProcessor, CLIPModel

# 1. Setup Environment
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "ask_textile", ".env"))
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    print("Error: Missing PINECONE_API_KEY")
    exit(1)

# 2. Setup Pinecone Indexes
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define our indexes
TEXT_INDEX = "website-text"
IMAGE_INDEX = "website-images"

existing_indexes = pc.list_indexes().names()

if TEXT_INDEX not in existing_indexes:
    print(f"Creating Pinecone index: {TEXT_INDEX} (dim 768 for nomic-embed-text)")
    pc.create_index(
        name=TEXT_INDEX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

if IMAGE_INDEX not in existing_indexes:
    print(f"Creating Pinecone index: {IMAGE_INDEX} (dim 512 for CLIP)")
    pc.create_index(
        name=IMAGE_INDEX,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Wait for indexes to be ready
def wait_index(name):
    while not pc.describe_index(name).status["ready"]:
        time.sleep(1)

wait_index(TEXT_INDEX)
wait_index(IMAGE_INDEX)

text_idx = pc.Index(TEXT_INDEX)
image_idx = pc.Index(IMAGE_INDEX)

# 3. Setup Embedding Models
print("Loading Text Embeddings (nomic-embed-text)...")
text_embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("Loading Image Embeddings (CLIP)...")
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

def get_clip_embedding(image_path=None):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    image_features = clip_model.get_image_features(**inputs)
    return image_features.detach().numpy()[0].tolist()

# 4. Extract Data from PDFs
BASE_DIR = os.path.dirname(__file__)
STATIC_IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)

pdf_files = [f for f in os.listdir(BASE_DIR) if f.endswith(".pdf")]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

all_text_chunks = []   # [(text, metadata), ...]
all_images = []        # [(image_path, metadata), ...]

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
                    "id": f"doc_{pdf_idx}_pg_{page_num}_ch_{i}",
                    "text": chunk,
                    "metadata": {
                        "project": project_name,
                        "page": page_num + 1,
                        "text": chunk  # We store the text in metadata so RAG can retrieve it!
                    }
                })
        
        # Extract Images
        blocks = page.get_text("blocks")
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            
            # Find image location for context extraction
            img_rects = page.get_image_rects(xref)
            y0, y1 = (img_rects[0].y0, img_rects[0].y1) if img_rects else (0, 0)
            
            # Search for the best text block to use as a description
            candidates = []
            for block in blocks:
                text = block[4].strip()
                # Skip very short text (likely page numbers or garbage) or purely vertical text
                if len(text) < 10 or block[2] - block[0] < 5: 
                    continue
                
                # Check distance from image
                dist_y = min(abs(block[1] - y1), abs(block[3] - y0))
                if dist_y < 120:
                    # Score based on proximity and length
                    score = (120 - dist_y) + (len(text) / 2)
                    candidates.append((score, text))
            
            # Use the highest scoring candidate
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                description = candidates[0][1].replace("\n", " ")
            else:
                description = "Textile project illustration"
            
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            
            # Save to disk
            img_filename = f"{project_name}_pg{page_num+1}_img{img_idx}.{ext}".replace(" ", "_")
            img_path = os.path.join(STATIC_IMAGES_DIR, img_filename)
            
            with open(img_path, "wb") as f:
                f.write(image_bytes)
                
            all_images.append({
                "id": f"img_{pdf_idx}_{page_num}_{img_idx}",
                "image_path": img_path,
                "metadata": {
                    "project": project_name,
                    "page": page_num + 1,
                    "image_url": f"/static/images/{img_filename}",
                    "description": description
                }
            })

# 5. Upload to Pinecone
print(f"Found {len(all_text_chunks)} text chunks. Uploading to Pinecone...")
batch_size = 50

# Text chunks
for i in range(0, len(all_text_chunks), batch_size):
    batch = all_text_chunks[i:i+batch_size]
    
    # Generate embeddings
    texts = [item["text"] for item in batch]
    vectors = text_embeddings.embed_documents(texts)
    
    to_upsert = []
    for j in range(len(batch)):
        to_upsert.append((batch[j]["id"], vectors[j], batch[j]["metadata"]))
        
    text_idx.upsert(vectors=to_upsert)
    print(f"  -> Uploaded text batch {i} to {i+len(batch)}")

print(f"Found {len(all_images)} images. Uploading to Pinecone...")
for i in range(0, len(all_images), batch_size):
    batch = all_images[i:i+batch_size]
    
    to_upsert = []
    for item in batch:
        vec = get_clip_embedding(item["image_path"])
        to_upsert.append((item["id"], vec, item["metadata"]))
        
    image_idx.upsert(vectors=to_upsert)
    print(f"  -> Uploaded image batch {i} to {i+len(batch)}")

print("Fully ingested PDFs into Pinecone and extracted images locally!")
