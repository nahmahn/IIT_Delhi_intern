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
# website-text-v2: High-resolution BGE embeddings
text_idx = pc.Index("website-text-v2")
# website-images-text: Reuse existing image-text index
image_idx = pc.Index("website-images-text")

# Load models
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

# --- DYNAMIC PROJECT DISCOVERY ---
@functools.lru_cache(maxsize=1)
def get_all_projects_map():
    """Create a map of keywords to actual project filenames."""
    projs = [f.replace(".pdf", "") for f in os.listdir(BASE_DIR) if f.endswith(".pdf")]
    mapping = {}
    for p in projs:
        # Extract keywords from filename (e.g., 'Baluchari', 'Muslin')
        # We take the first part before the first underscore or space
        keyword = p.split("_")[0].split(" ")[0].lower()
        mapping[keyword] = p
    return mapping, projs

def get_projects_from_text(text):
    """Fallback: detect projects by simple keyword matching in text."""
    mapping, all_projs = get_all_projects_map()
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
    """Retrieve text chunks, then re-rank with Cross-Encoder."""
    # 1. Native Pinecone Filter (Strict discovery)
    # Ensure we use keyword fallback if LLM missed it
    extra_detected = get_projects_from_text(query)
    final_filters = list(set((detected_projects or []) + extra_detected))
    
    filter_dict = None
    if final_filters:
        filter_dict = {"project": {"$in": final_filters}}
        print(f"RAG: Final text filter applied -> {final_filters}")

    # Fetch 15 candidates to rerank
    results = text_idx.query(
        vector=query_vec, 
        top_k=15, 
        include_metadata=True,
        filter=filter_dict
    )
    
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
    """Retrieve images. Only performs a balanced split if it's a clear comparison."""
    # Safety: Ensure keyword-based projects are included (LLM fallback)
    extra_detected = get_projects_from_text(query)
    final_filters = list(set((project_filters or []) + extra_detected))

    # 1. No filter or single project -> Standard search across the group
    if not final_filters or (len(final_filters) == 1 and not is_comparison):
        filter_dict = None
        if final_filters:
            filter_dict = {"project": {"$in": final_filters}}
            print(f"RAG: Image filter applied -> {final_filters}")
            
        results = image_idx.query(vector=query_vec, top_k=top_k, include_metadata=True, filter=filter_dict)
        return _parse_image_results(results)

    # 2. Comparison detected -> Fetch separately to ensure balance
    num_projects = len(final_filters)
    per_project_k = max(2, top_k // num_projects)
    print(f"RAG: Comparison query confirmed ({num_projects} projects). Balanced image fetch active.")

    all_matches = []
    for project in final_filters:
        res = image_idx.query(
            vector=query_vec, 
            top_k=per_project_k, 
            include_metadata=True,
            filter={"project": {"$eq": project}}
        )
        all_matches.extend(res["matches"])
    
    all_matches.sort(key=lambda x: x["score"], reverse=True)
    return _parse_image_results({"matches": all_matches})

def _parse_image_results(results):
    """Helper to parse Pinecone matches into our app format."""
    images = []
    for match in results.get("matches", []):
        md = match["metadata"]
        images.append({
            "url": md.get("image_url"),
            "project": md.get("project"),
            "page": md.get("page"),
            "description": md.get("description", ""),
            "score": round(match["score"], 3)
        })
    return images

# ============================================================
# GUARDRAIL
# ============================================================
def is_query_relevant(query):
    """Safety check: ensures query is related to textiles/department."""
    q = query.lower()
    # Dynamic guardrails: includes words from all project names
    dynamic_keywords = []
    _, all_projs = get_all_projects_map()
    for proj in all_projs:
        dynamic_keywords.extend(proj.lower().split("_"))
    
    extra_keywords = [
        "saree", "sari", "textile", "weaving", "loom", "fabric", "cotton", "silk",
        "design", "pattern", "motif", "artisan", "handloom", "dyeing", "yarn",
        "warp", "weft", "project", "department", "dept", "findings", "collection",
        "data", "report", "professor", "dr", "prof", "dst", "shri",
        "साड़ी", "बुनाई", "कपास", "रेशम", "डिज़ाइन", "परियोजना",
        "बालूचरी", "मसलिन", "नेगमम", "फुलकारी",
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

    system_msg = f"""You are a helpful assistant for the Textile Department. 
    You ONLY answer questions about these 4 textile projects: Baluchari, Muslin, Negamam, and Phulkari.
    
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
    
    _, all_projs = get_all_projects_map()
    
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
       - IMPORTANT: If the question is about 'Muslin', you MUST return 'Muslin_Dr. Shantanu Basak...'.
       - IMPORTANT: If the question is about 'Baluchari', you MUST return 'Baluchari Saree_Prof. Wazed Ali...'.
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
    
    # 0. Quick Guardrail
    if not is_query_relevant(query):
        return {
            "answer": "I'm sorry, but I can only assist with questions related to the Textile Department projects (Baluchari, Muslin, Negamam, etc.). How can I help you with those?",
            "images": [],
            "sources": []
        }

    # 1. LLM-Powered Rewriting and Project Detection
    t0 = time.time()
    standalone_query, detected_projects, is_comp = get_standalone_query_and_projects(query, chat_history)
    print(f"  >> Step 1 (LLM rewrite):  {time.time()-t0:.2f}s")
    
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
