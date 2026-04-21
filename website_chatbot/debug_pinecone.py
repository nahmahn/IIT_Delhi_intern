import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(".env")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

for idx_name in ["website-text-v4", "website-images-v4"]:
    idx = pc.Index(idx_name)
    stats = idx.describe_index_stats()
    print(f"\nIndex: {idx_name}")
    print(f"Total Vectors: {stats.total_vector_count}")
    
    if stats.total_vector_count > 0:
        # Get one sample to see metadata structure
        # Query for something very common (or just any vector)
        res = idx.query(vector=[0.0]*768, top_k=1, include_metadata=True)
        if res.matches:
            print(f"Sample Project: {res.matches[0].metadata.get('project')}")
            print(f"Sample Metadata: {res.matches[0].metadata.keys()}")
