"""
Re-caption all images using Groq LLM.
Strategy:
  1. For each PDF, extract text from each page
  2. For each image on that page, ask Groq to generate a 1-2 sentence caption
     based on the page text + image position
  3. Update the Pinecone metadata (description field) without re-embedding
"""
import fitz
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq

# Setup
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "ask_textile", ".env"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
image_idx = pc.Index("website-images")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

BASE_DIR = os.path.dirname(__file__)
pdf_files = [f for f in os.listdir(BASE_DIR) if f.endswith(".pdf")]

def generate_caption(page_text, project_name, page_num, img_index):
    """Use Groq to generate a proper caption from page context."""
    # Trim page text to avoid token limits
    page_text = page_text[:2000]
    
    prompt = f"""You are looking at page {page_num} of a document about the project: "{project_name}".
This page contains image #{img_index + 1}. Based on the page text below, write a SHORT, specific caption (1-2 sentences max) describing what this image most likely shows.

Rules:
- Be specific to the project (mention saree type, technique, etc.)
- If the page mentions figures, diagrams, or photos, use that info
- If unsure, describe based on the project name and page content
- Keep it under 30 words

Page text:
{page_text}

Caption:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=60,
        )
        caption = response.choices[0].message.content.strip().strip('"')
        return caption
    except Exception as e:
        print(f"  Error generating caption: {e}")
        time.sleep(2)  # Rate limit cooldown
        return f"Image from {project_name} project, page {page_num}"

# Process each PDF and generate captions
updates = []  # (pinecone_id, new_description)

for pdf_idx, pdf_file in enumerate(pdf_files):
    pdf_path = os.path.join(BASE_DIR, pdf_file)
    project_name = pdf_file.replace(".pdf", "")
    print(f"\n=== Processing: {project_name} ===")
    
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        images = page.get_images(full=True)
        
        if not images:
            continue
            
        for img_idx, img in enumerate(images):
            pinecone_id = f"img_{pdf_idx}_{page_num}_{img_idx}"
            
            # Generate caption using LLM
            caption = generate_caption(page_text, project_name, page_num + 1, img_idx)
            updates.append((pinecone_id, caption))
            
            print(f"  [{pinecone_id}] {caption}")
            
            # Small delay to avoid Groq rate limits
            time.sleep(0.3)

# Now update Pinecone metadata in batches
print(f"\n\n=== Updating {len(updates)} image captions in Pinecone ===")

# We need to fetch existing vectors, update metadata, and upsert back
batch_size = 20
for i in range(0, len(updates), batch_size):
    batch = updates[i:i+batch_size]
    ids = [u[0] for u in batch]
    
    # Fetch existing vectors (we need the embeddings)
    fetched = image_idx.fetch(ids=ids)
    
    upsert_batch = []
    for pid, new_desc in batch:
        if pid in fetched.vectors:
            vec = fetched.vectors[pid]
            md = vec.metadata
            md["description"] = new_desc
            upsert_batch.append((pid, vec.values, md))
    
    if upsert_batch:
        image_idx.upsert(vectors=upsert_batch)
        print(f"  Updated batch {i} to {i+len(batch)}")

print("\nDone! All image captions have been updated in Pinecone.")
