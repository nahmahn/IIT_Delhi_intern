import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)
index_name = "ask-textile"

print(f"Checking index: {index_name}")
try:
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print("Connection Successful!")
    print(f"Index Stats: {stats}")
except Exception as e:
    print(f"Error: {e}")
