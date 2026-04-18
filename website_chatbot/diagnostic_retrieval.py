import os
from rag import text_embeddings, text_idx, detect_project_from_query

query = "how is baluchari made"
print(f"--- Diagnostic for: {query} ---")

# 1. Detect
detected = detect_project_from_query(query)
print(f"Detected: {detected}")

# 2. Raw Embed Query
vec = text_embeddings.embed_query(query)
results = text_idx.query(vector=vec, top_k=10, include_metadata=True)

print("\nRaw Results (Before Boost):")
for i, match in enumerate(results["matches"]):
    print(f"{i+1}. {match['metadata']['project']} | Score: {match['score']:.4f}")

print("\nBoosted Results:")
scored = []
for match in results["matches"]:
    md = match["metadata"]
    project = md.get("project")
    score = match["score"]
    
    boost = 0.0
    if detected and any(d.lower() == project.lower() for d in detected):
        boost = 0.5 # Testing +0.5 boost
        score += boost
    
    scored.append((project, score, boost))

scored.sort(key=lambda x: x[1], reverse=True)
for i, (p, s, b) in enumerate(scored):
    print(f"{i+1}. {p} | Total: {s:.4f} (Boost: {b})")
