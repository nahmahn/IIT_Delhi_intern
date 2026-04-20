import os
import time
from collections import Counter
import functools
import concurrent.futures
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from groq import Groq
import torch

# Load environment
BASE_DIR = os.path.dirname(__file__)
if os.path.exists(os.path.join(BASE_DIR, ".env")):
    load_dotenv(os.path.join(BASE_DIR, ".env"))
else:
    load_dotenv(os.path.join(BASE_DIR, "..", "ask_textile", ".env"))

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
# website-text-v3: Professional metadata-driven index
text_idx = pc.Index("website-text-v3")
# website-images-v3: Professional metadata-driven image index
image_idx = pc.Index("website-images-v3")

# Load models - Forcing CPU to avoid local DLL crashes/driver issues
# Dynamic device selection (CUDA for local, CPU for Hugging Face free tier)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"RAG: Loading Local Embeddings ({device})...")
text_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)

print(f"RAG: Loading Local Reranker ({device})...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

print("RAG: Using Groq Llama 70B Versatile...")
groq_client = Groq(api_key=GROQ_API_KEY)

# --- EMBEDDING CACHE ---
_embedding_cache = {}
def cached_embed(query):
    """Cache high-resolution BGE embeddings."""
    if query in _embedding_cache:
        return _embedding_cache[query]
    
    # Instruction for BGE-base retrieval
    instruction = "represent the query for retrieval: "
    vec = text_model.encode(instruction + query).tolist()
    
    _embedding_cache[query] = vec
    if len(_embedding_cache) > 200:
        oldest = next(iter(_embedding_cache))
        del _embedding_cache[oldest]
    return vec

# --- METADATA CLASSIFICATION RULES (Aligned with Ingest) ---
DOC_TYPE_RULES = {
    "project_report": ["baluchari", "muslin", "negamam", "phulkari", "maheshwari"],
    "dept_info": ["shri", "centre", "department"],
}

# --- DYNAMIC PROJECT DISCOVERY ---
@functools.lru_cache(maxsize=1)
def get_all_projects_map():
    """Create a map of keywords to actual project filenames for query rewriting."""
    projs = [f.replace(".pdf", "") for f in os.listdir(BASE_DIR) if f.endswith(".pdf")]
    mapping = {}
    
    for p in projs:
        name_lower = p.lower()
        # Map first word/keyword to full local name
        keyword = p.split("_")[0].split(" ")[0].lower()
        mapping[keyword] = p
            
    return mapping, projs 

def get_projects_from_text(text):
    """Fallback: detect project names by simple keyword matching."""
    mapping, _ = get_all_projects_map()
    detected = []
    text_lower = text.lower()
    for kw, full_name in mapping.items():
        if kw in text_lower:
            detected.append(full_name)
    return list(set(detected))


# ============================================================
# HIGH-PRECISION RETRIEVAL — Semantic Search + Local Reranking
# ============================================================
def retrieve_text(query, query_vec, detected_projects=None, top_k=3):
    """Retrieve text chunks using doc_type metadata filtering."""
    extra_detected = get_projects_from_text(query)
    final_projects = list(set((detected_projects or []) + extra_detected))
    
    # Text Retrieval Filter Strategy:
    # 1. Detected Projects -> Focus on those + Dept Info + Supplementary
    # 2. No Projects -> Global search (No filter)
    if final_projects:
        filter_dict = {
            "$or": [
                {"project": {"$in": final_projects}},
                {"doc_type": {"$in": ["dept_info", "supplementary"]}}
            ]
        }
        print(f"RAG: Text filter active (Project + Dept + Supp) -> {final_projects}")
    else:
        filter_dict = None
        print("RAG: Global text search active (No filters).")

    # Fetch candidates for reranking
    results = text_idx.query(
        vector=query_vec, 
        top_k=30, 
        include_metadata=True,
        filter=filter_dict
    )

    
    if not results["matches"]:
        # Fallback: if filtered search produced nothing, try a global search
        results = text_idx.query(vector=query_vec, top_k=30, include_metadata=True)
        if not results["matches"]:
            return "No relevant context found.", []

    # 2. Local Cross-Encoder Reranking (HD Quality)
    candidates = []
    for match in results["matches"]:
        md = match["metadata"]
        candidates.append({
            "text": md.get("text", ""),
            "project": md.get("project", "Unknown"),
            "page": md.get("page", 1),
            "original_score": match["score"]
        })

    # Prepare pairs for cross-encoder: (query, text)
    rerank_pairs = [(query, c['text']) for c in candidates]
    rerank_scores = reranker.predict(rerank_pairs)

    for i, score in enumerate(rerank_scores):
        candidates[i]['rerank_score'] = score

    # Sort by rerank score (Higher is better)
    candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    top_candidates = candidates[:top_k]
    
    contexts = []
    sources = []
    for item in top_candidates:
        contexts.append(f"[{item['project']}, Page {item['page']}]: {item['text']}")
        sources.append({"project": item["project"], "page": item["page"]})
    
    return "\n\n".join(contexts), sources

def retrieve_images(query, query_vec, top_k=6, project_filters=None, is_comparison=False):
    """Retrieve images using metadata-driven doc_type filtering."""
    extra_detected = get_projects_from_text(query)
    final_projects = list(set((project_filters or []) + extra_detected))

    SCORE_THRESHOLD = 0.55  # hard filter for low relevance

    # Image Retrieval Filter Strategy (Mentor-aligned):
    # 1. Projects Detected -> Only fetch those specific projects + dept_info
    # 2. No Projects -> Fetch project_reports + dept_info (Exclude supplementary/charts)
    if final_projects:
        filter_dict = {
            "$or": [
                {"project": {"$in": final_projects}},
                {"doc_type": {"$eq": "dept_info"}}
            ]
        }
        print(f"RAG: Image filter active (Project + Dept) -> {final_projects}")
    else:
        filter_dict = {
            "doc_type": {"$in": ["project_report", "dept_info"]}
        }
        print("RAG: Global image filter active (Project Reports + Dept Info).")

    if not is_comparison:
        results = image_idx.query(
            vector=query_vec, 
            top_k=30,  # Fetch more for reranking
            include_metadata=True, 
            filter=filter_dict
        )
    else:
        # Comparison mode: fetch separately from each project to ensure balance
        per_project_k = max(4, (top_k * 2) // max(len(final_projects), 1))
        all_matches = []
        for project in final_projects:
            res = image_idx.query(
                vector=query_vec, 
                top_k=per_project_k, 
                include_metadata=True,
                filter={"project": {"$eq": project}}
            )
            all_matches.extend(res["matches"])
        
        results = {"matches": sorted(all_matches, key=lambda x: x["score"], reverse=True)}

    # 1. Hard Filter and Candidate Harvesting
    candidates = []
    for match in results.get("matches", []):
        if match["score"] < SCORE_THRESHOLD:
            continue
            
        md = match["metadata"]
        candidates.append({
            "description": md.get("description", "").strip(),
            "project": md.get("project"),
            "page": md.get("page"),
            "image_url": md.get("image_url", ""),
            "score": match["score"]
        })

    if not candidates:
        print("RAG: No image candidates passed the score threshold.")
        return []

    # 2. Cross-Encoder Reranking
    rerank_pairs = [(query, c["description"]) for c in candidates]
    rerank_scores = reranker.predict(rerank_pairs)
    for i, score in enumerate(rerank_scores):
        candidates[i]["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # 3. Deduplication and Diversity (Max 2 images per project)
    seen_desc = set()
    project_count = Counter()
    final_images = []

    for c in candidates:
        if len(final_images) >= top_k:
            break
        
        if c["description"] in seen_desc:
            continue
            
        if project_count[c["project"]] >= 2 and not is_comparison:
            continue
            
        seen_desc.add(c["description"])
        project_count[c["project"]] += 1

        filename = os.path.basename(c["image_url"])
        final_images.append({
            "url": f"/static/images/{filename}",
            "project": c["project"],
            "page": c["page"],
            "description": c["description"],
            "score": round(c["rerank_score"], 3)
        })

    print(f"RAG: Final image results -> {len(final_images)} items")
    return final_images

# ============================================================
# GUARDRAIL
# ============================================================
def is_query_relevant(query):
    """Safety check: ensures query is related to textiles/department."""
    q = query.lower()
    # Dynamic guardrails: includes words from all project names
    dynamic_keywords = []
    mapping, all_projs = get_all_projects_map()
    for proj in all_projs:
        dynamic_keywords.extend(proj.lower().split("_"))
    
    extra_keywords = [
        "saree", "sari", "textile", "weaving", "loom", "fabric", "cotton", "silk",
        "design", "pattern", "motif", "artisan", "handloom", "dyeing", "yarn",
        "warp", "weft", "project", "department", "dept", "findings", "collection",
        "data", "report", "professor", "dr", "prof", "dst", "shri",
        "साड़ी", "बुनाई", "कपास", "रेशम", "डिज़ाइन", "परियोजना",
        "बालूचरी", "मसलिन", "नेगमम", "फुलकारी",
        "baluchari", "muslin", "negamam", "phulkari", "maheshwari",
        "history", "origin", "cluster", "weaver",
        "hello", "hi", "who are you", "who are u", "assistant", "help"
    ]
    
    for kw in set(dynamic_keywords + extra_keywords):
        if kw and len(kw) > 2 and kw in q:
            return True
            
    return False

# ============================================================
# CONVERSATION MEMORY
# ============================================================
chat_history = []

def generate_answer(query, context, images_context, language="English"):
    lang_instruction = f"Respond in {language} language." if language.lower() not in ["english", "en"] else ""

    images_info = "\n".join([
        f"- Image {i+1}: {img['description'][:150]} (from {img['project']}, Page {img['page']})" 
        for i, img in enumerate(images_context)
    ])

    mapping, all_projs = get_all_projects_map()
    project_list = ", ".join(all_projs)

    system_msg = f"""You are a helpful assistant for the Textile Department. 
    You ONLY answer questions about these projects: {project_list}.
    
    STRICT RULES:
    1. Answer ONLY using the context provided below.
    2. If the context does not contain the answer, say "Main is baare mein jaankari nahi de sakta, mera focus sirf Textile Department ke project reports par hai." (I cannot provide information on that, my focus is only on Textile Department project reports.)
    3. If the user asks who you are, say "I am a research assistant for the Textile Department."
    4. DO NOT answer general questions (math, recipes, AI models, etc.).

    Context:
    {context}
    
    Related Images:
    {images_info}
    
    {lang_instruction}"""

    messages = [{"role": "system", "content": system_msg}]
    messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": query})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content
    
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})
    
    return answer

def get_standalone_query_and_projects(query, history):
    """Ask LLM to rewrite query AND identify projects involved."""
    history_str = "\n".join([f"{m['role']}: {m['content'][:300]}" for m in history[-4:]])
    
    mapping, all_projs = get_all_projects_map()
    
    # We ask for a JSON response to keep it clean
    prompt = f"""You are an expert textile researcher. Analyze the follow-up question in context of history.
    
    Available Project Files:
    {", ".join(all_projs)}
    
    Recent Chat History:
    {history_str}
    
    Current Follow-up: {query}
    
    TASK:
    1. Rephrase as a standalone search query (Concepts ONLY, remove professor names/dates).
    2. Identify ONLY the project names from the list above that are DIRECTLY RELEVANT.
       - IMPORTANT: You MUST return exact string matches to the items in "Available Project Files" (e.g. including the professor and date suffixes if they exist in the list). Do not truncate the names.
    3. Determine if this is a comparison query between multiple projects.
    
    Response format:
    Query: [Semantic query]
    Projects: [Comma separated list of EXACT project names]
    IsComparison: [True/False]"""

    # Use fast 8B model for lightweight rewriting (70B is overkill here)
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    res = response.choices[0].message.content
    
    # Parse lines
    rewritten = query
    llm_projects = []
    is_comparison = False
    for line in res.split("\n"):
        if line.startswith("Query:"):
            rewritten = line.replace("Query:", "").strip()
        if line.startswith("Projects:"):
            llm_projects = [p.strip().lower() for p in line.replace("Projects:", "").split(",") if p.strip()]
        if line.startswith("IsComparison:"):
            is_comparison = "true" in line.lower()
            
    # Final normalization
    final_projects = []
    for lp in llm_projects:
        for ap in all_projs:
            if lp in ap.lower():
                final_projects.append(ap)
                
    print(f"RAG: LLM Rewritten -> {rewritten}")
    print(f"RAG: LLM Projects  -> {list(set(final_projects))} (Comparison: {is_comparison})")
    return rewritten, list(set(final_projects)), is_comparison

# ============================================================
# MAIN QUERY HANDLER
# ============================================================
# --- MAIN QUERY HANDLER ---
def process_query(query, language="English"):
    t_total = time.time()
    
    # 1. LLM-Powered Rewriting and Project Detection
    t0 = time.time()
    standalone_query, detected_projects, is_comp = get_standalone_query_and_projects(query, chat_history)
    print(f"  >> Step 1 (LLM rewrite):  {time.time()-t0:.2f}s")

    # 0. Quick Guardrail (Now checking the rewritten standalone query!)
    if not is_query_relevant(standalone_query):
        print("  >> Guardrail blocked:", standalone_query)
        return {
            "answer": "I'm sorry, but I can only assist with questions related to the Textile Department projects (Baluchari, Muslin, Negamam, etc.).",
            "images": [],
            "sources": []
        }
    
    # 2. Embed the query exactly ONCE with cache
    t0 = time.time()
    query_vec = cached_embed(standalone_query)
    print(f"  >> Step 2 (Embedding):    {time.time()-t0:.2f}s")
    
    # 3. Retrieve Text and Images concurrently
    t0 = time.time()
    img_k = 10 if is_comp else 6
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_text = executor.submit(retrieve_text, standalone_query, query_vec, detected_projects)
        future_images = executor.submit(retrieve_images, standalone_query, query_vec, img_k, detected_projects, is_comp)
        
        context, sources = future_text.result()
        images = future_images.result()
    print(f"  >> Step 3 (Pinecone + Rerank): {time.time()-t0:.2f}s")
    
    # 4. Generate answer with memory
    t0 = time.time()
    answer = generate_answer(query, context, images, language)
    print(f"  >> Step 4 (LLM answer):   {time.time()-t0:.2f}s")
    
    print(f"  == TOTAL: {time.time()-t_total:.2f}s")
    
    return {
        "answer": answer,
        "images": images,
        "sources": sources
    }
