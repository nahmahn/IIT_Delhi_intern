import os
from dotenv import load_dotenv
from rag import get_standalone_query_and_projects, get_all_projects

load_dotenv(".env")

query = "What are the main clusters for Negamam cotton sarees?"
rewritten, detected, is_comp = get_standalone_query_and_projects(query, [])

print(f"Query: {query}")
print(f"All Projects available: {get_all_projects()}")
print(f"Detected: {detected}")

if not detected:
    q_lower = query.lower()
    for ap in get_all_projects():
        core_id = ap.replace("_", " ").split(" ")[0].lower()
        print(f"Checking '{core_id}' in '{q_lower}' -> {core_id in q_lower}")
