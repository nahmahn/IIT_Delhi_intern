"""
Migrate from CLIP-based image retrieval to Text-based image retrieval.
We fetch all images, extract their high-quality LLM captions, embed those
captions with `nomic-embed-text`, and save them to a NEW index `website-images-text`.
"""
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OllamaEmbeddings

# 1. Setup
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "ask_textile", ".env"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

text_embeddings = OllamaEmbeddings(model="nomic-embed-text")

NEW_INDEX = "website-images-text"

# 2. Check or create new index (dim 768 for nomic)
existing_indexes = pc.list_indexes().names()
if NEW_INDEX not in existing_indexes:
    print(f"Creating Pinecone index: {NEW_INDEX}")
    pc.create_index(
        name=NEW_INDEX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(NEW_INDEX).status["ready"]:
        time.sleep(1)

new_image_idx = pc.Index(NEW_INDEX)
old_image_idx = pc.Index("website-images")

# 3. Fetch all current image metadata from old index
print("Fetching all image metadata from old index...")
all_ids = []
for pdf in range(5):
    for pg in range(50): # assume max 50 pages
        for im in range(10): # assume max 10 images per page
            all_ids.append(f"img_{pdf}_{pg}_{im}")

# fetch in chunks of 500
fetched_items = []
for i in range(0, len(all_ids), 500):
    chunk_ids = all_ids[i:i+500]
    res = old_image_idx.fetch(ids=chunk_ids)
    for k, v in res.vectors.items():
        fetched_items.append({"id": k, "metadata": v.metadata})

print(f"Fetched {len(fetched_items)} image records.")

if not fetched_items:
    print("No items fetched... check IDs")
else:
    # 4. Embed captions and upload to new index
    count = 0
    for item in fetched_items:
        caption = item["metadata"].get("description", "Image")
        
        # Embed with Nomic
        try:
            vector = text_embeddings.embed_query(caption)
            new_image_idx.upsert(vectors=[(item["id"], vector, item["metadata"])])
            count += 1
            print(f"Upserted {count}/{len(fetched_items)}: {item['id']}")
        except Exception as e:
            print(f"Error upserting {item['id']}: {e}")

print("Migration to text-based image search complete!")

