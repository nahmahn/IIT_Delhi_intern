import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment
load_dotenv(os.path.join("website_chatbot", ".env"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

idx = pc.Index("website-images-v4")

print("--- Pinecone Image Audit ---")
stats = idx.describe_index_stats()
print(f"Stats: {stats}")

# Fetch a few records from the index
results = idx.query(
    vector=[0.1]*768, # dummy vector
    top_k=5,
    include_metadata=True
)

for i, match in enumerate(results["matches"]):
    md = match["metadata"]
    print(f"\nMatch {i+1}:")
    print(f"  ID: {match['id']}")
    print(f"  Metadata Project:   {md.get('project')}")
    print(f"  Metadata Image URL: {md.get('image_url')}")
    print(f"  Description:        {md.get('description', '')[:100]}...")
    # Check if they mismatch
    proj_name = md.get('project', '').lower()
    url_name = md.get('image_url', '').lower()
    if ('baluchari' in proj_name and 'baluchari' not in url_name) or \
       ('negamam' in proj_name and 'negamam' not in url_name) or \
       ('muslin' in proj_name and 'muslin' not in url_name):
        print("  !!! MISMATCH DETECTED !!!")
