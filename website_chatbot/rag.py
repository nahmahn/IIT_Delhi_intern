import os
import time
from collections import Counter
import functools
import concurrent.futures
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
import ollama
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
# website-text-v4: Professional metadata-driven index
text_idx = pc.Index("website-text-v4")
# website-images-v4: Professional metadata-driven image index
image_idx = pc.Index("website-images-v4")

# Load models - Forcing CPU to avoid local DLL crashes/driver issues
# Dynamic device selection (CUDA for local, CPU for Hugging Face free tier)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"RAG: Loading Local Embeddings ({device})...")
text_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)

print(f"RAG: Loading Local Reranker ({device})...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

print("RAG: Using Local Ollama (llama3.1:8b)...")
# groq_client = Groq(api_key=GROQ_API_KEY)

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
def get_all_projects():
    """Fetch unique project names directly from Pinecone — no filesystem dependency."""
    try:
        # Query with a dummy vector to get metadata from recent records
        dummy_vec = [0.0] * 768
        results = text_idx.query(vector=dummy_vec, top_k=100, include_metadata=True)
        projects = list({m["metadata"]["project"] for m in results["matches"]})
        if not projects:
            # Fallback to filesystem if index is empty/initializing
            return [f.replace(".pdf", "") for f in os.listdir(BASE_DIR) if f.endswith(".pdf")]
        return projects
    except Exception as e:
        print(f"RAG: Discovery error: {e}")
        return [f.replace(".pdf", "") for f in os.listdir(BASE_DIR) if f.endswith(".pdf")]

def get_projects_from_text(text):
    """Detect project names by keyword matching against Pinecone-discovered list."""
    all_projects = get_all_projects()
    detected = []
    text_lower = text.lower()
    for p in all_projects:
        # Check if project name or its first word is in query
        keyword = p.split("_")[0].split(" ")[0].lower()
        if keyword in text_lower or p.lower() in text_lower:
            detected.append(p)
    return list(set(detected))


# ============================================================
# HIGH-PRECISION RETRIEVAL — Semantic Search + Local Reranking
# ============================================================
def retrieve_text(query, query_vec, detected_projects=None, top_k=3):
    """Retrieve text chunks using doc_type metadata filtering."""
    extra_detected = get_projects_from_text(query)
    final_projects = list(set((detected_projects or []) + extra_detected))
    
    # Text Retrieval Filter Strategy:
    # 1. Detected Projects -> Focus on those + Dept Info + Supplementary (conditional)
    # 2. No Projects -> Global search (No filter)
    if final_projects:
        or_conditions = [
            {"project": {"$in": final_projects}},
            {"doc_type": {"$eq": "dept_info"}}
        ]
        
        # Include supplementary only for cross-cutting queries (mentor feedback rule)
        supp_keywords = ["carbon", "footprint", "emission", "environment", "sustainability", "heritage products", "location"]
        if any(kw in query.lower() for kw in supp_keywords):
            or_conditions.append({"doc_type": {"$eq": "supplementary"}})
            
        filter_dict = {"$or": or_conditions}
        print(f"RAG: Text filter active -> {final_projects} (+ Dept/Supp if relevant)")
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
        clean_name = get_clean_project_name(item['project'])
        contexts.append(f"[Source: {clean_name}, Page {item['page']}]: {item['text']}")
        sources.append({"project": item["project"], "page": item["page"]})
    
    return "\n\n".join(contexts), sources

def get_clean_project_name(raw_name):
    """Strip professor names, dates, codes from project names for cleaner answers."""
    parts = raw_name.split("_")
    clean_parts = []
    for p in parts:
        # Stop at technical suffixes
        if any(p.startswith(prefix) for prefix in ["Dr.", "Prof.", "DST", "Data"]):
            break
        clean_parts.append(p)
    return " ".join(clean_parts) if clean_parts else parts[0]

def retrieve_images(query, query_vec, top_k=6, project_filters=None, is_comparison=False):
    """Retrieve images using metadata-driven doc_type filtering."""
    extra_detected = get_projects_from_text(query)
    final_projects = list(set((project_filters or []) + extra_detected))

    SCORE_THRESHOLD = 0.55  # hard filter for low relevance

    # Image Retrieval Filter Strategy (Mentor-aligned):
    # 1. Projects Detected -> Only fetch those specific projects (strict, no leakage)
    # 2. No Projects -> Fetch project_reports + dept_info (Exclude supplementary/charts)
    if final_projects:
        filter_dict = {"project": {"$in": final_projects}}
        print(f"RAG: Image filter strict (Project only) -> {final_projects}")
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

        clean_proj = get_clean_project_name(c["project"])
        filename = os.path.basename(c["image_url"])
        final_images.append({
            "url": f"/static/images/{filename}",
            "project": c["project"],
            "page": c["page"],
            "description": c["description"],
            "clean_project": clean_proj,
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
    all_projs = get_all_projects()
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

def generate_answer(query, context, images_context, language="English", is_comparison=False):
    lang_instruction = f"Respond in {language} language." if language.lower() not in ["english", "en"] else ""

    images_info = "\n".join([
        f"- Image {i+1}: {img['description'][:150]} (Source: {img['clean_project']})" 
        for i, img in enumerate(images_context)
    ])

    all_projs = get_all_projects()
    project_list = ", ".join(all_projs)

    system_msg = f"""You are a concise research assistant for the Textile Department.
    
    STRICT RULES:
    1. Answer ONLY using the context provided below.
    2. Answer the SPECIFIC question asked. Do NOT provide a general overview.
    3. Keep answers to 3-4 sentences maximum, unless the user explicitly asks for detail.
    4. Use bullet points for lists.
    5. NEVER mention document filenames, report titles, professor names, or page numbers (like "Phulkari Designs_Dr. Priyanka..."). Use only clean names like "Phulkari".
    6. If the context does not contain the answer, say "I don't have specific information about this in the project reports."
    7. If the user asks who you are, say "I am a research assistant for the Textile Department."
    8. DO NOT answer general questions (math, recipes, AI models, etc.).

    Context:
    {context}
    
    Related Images:
    {images_info}
    
    {lang_instruction}"""

    messages = [{"role": "system", "content": system_msg}]
    messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": query})

    response = ollama.chat(
        model="llama3.1:8b",
        messages=messages,
        options={
            "temperature": 0.3,
            "num_predict": 1024,
        }
    )
    answer = response['message']['content']
    
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})
    
    return answer

def get_standalone_query_and_projects(query, history):
    """Ask LLM to rewrite query AND identify projects involved."""
    history_str = "\n".join([f"{m['role']}: {m['content'][:300]}" for m in history[-4:]])
    
    all_projs = get_all_projects()
    
    system_prompt = f"""You are an expert textile researcher. Your task is to rewrite follow-up questions into standalone search queries and identify RELEVANT projects.

AVAILABLE PROJECT FILES:
{", ".join(all_projs)}

TASK:
1. Rephrase the "CURRENT FOLLOW-UP" into a standalone search query (Query).
   - RECENCY RULE: If the user uses pronouns like "it", "this", or "they", they ALWAYS refer to the MOST RECENT textile or project mentioned in the history.
   - RELEVANCY RULE: If the "CURRENT FOLLOW-UP" is completely unrelated to textiles, research, or the department (e.g., food, recipes, sports, general math), DO NOT rewrite it to a textile query. Instead, set Query to "NOT_RELEVANT".
2. Identify projects from the list above that are DIRECTLY RELEVANT to the "CURRENT FOLLOW-UP" (Projects).
   - IMPORTANT: If the current query mentions a NEW textile, prioritize it. If it uses pronouns, use the most recent textile from history.

EXAMPLES:
- History: "About Muslin" -> Follow-up: "how is it made?" -> Query: "Manufacturing process of Muslin", Projects: [Muslin...], IsComparison: False
- History: "About Phulkari" -> Follow-up: "what about Baluchari?" -> Query: "Information about Baluchari", Projects: [Baluchari...], IsComparison: False
- History: "About Baluchari" -> Follow-up: "give me a pizza recipe" -> Query: "NOT_RELEVANT", Projects: [], IsComparison: False

RESPONSE FORMAT:
Query: [Semantic query]
Projects: [Comma separated list of EXACT project names]
IsComparison: [True/False]"""

    user_content = f"RECENT CHAT HISTORY:\n{history_str}\n\nCURRENT FOLLOW-UP: {query}"

    # Use local model for rewriting
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        options={   
            "temperature": 0,
            "num_predict": 150,
        }
    )
    res = response['message']['content']
    
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
    
    # 1. Normalize LLM output
    for lp in llm_projects:
        for ap in all_projs:
            # Multi-directional substring match
            if lp in ap.lower() or ap.lower() in lp:
                final_projects.append(ap)
                
    # 2. Safety Fallback: Keyword-based scan (Always run for current query to prevent LLM missing obvious keywords)
    q_lower = query.lower()
    for ap in all_projs:
        # Extract first word as core identifier (e.g., "Negamam", "Baluchari")
        core_id = ap.replace("_", " ").split(" ")[0].lower()
        if core_id in q_lower and ap not in final_projects:
            final_projects.append(ap)
                
    final_projects = list(set(final_projects))
    
    # 3. Hard Pruning: If a new textile is mentioned, purge unrelated projects from history
    if not is_comparison:
        core_textiles = ["baluchari", "muslin", "negamam", "phulkari", "maheshwari"]
        mentioned_textiles = [t for t in core_textiles if t in query.lower()]
        
        if mentioned_textiles:
            # Keep only projects that match the mentioned textiles
            pruned_projects = []
            for p in final_projects:
                p_lower = p.lower()
                # If this project belongs to one of the core textiles
                project_textile = next((t for t in core_textiles if t in p_lower), None)
                
                if project_textile:
                    # If it's the one we mentioned, keep it. If it's a DIFFERENT core textile, drop it.
                    if project_textile in mentioned_textiles:
                        pruned_projects.append(p)
                else:
                    # Keep non-textile projects (like "Carbon footprint...")
                    pruned_projects.append(p)
            final_projects = pruned_projects

    # Final check: If it was supposed to be a comparison but we only found one project...
    if is_comparison or len(final_projects) == 0:
        q_lower = query.lower()
        for ap in all_projs:
            core_id = ap.replace("_", " ").split(" ")[0].lower()
            if core_id in q_lower and ap not in final_projects:
                final_projects.append(ap)
                
    final_projects = list(set(final_projects))
    print(f"RAG: LLM Rewritten -> {rewritten}")
    print(f"RAG: Detected Projects -> {final_projects} (Comparison: {is_comparison})")
    
    return rewritten, final_projects, is_comparison

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

    # 0. Quick Guardrail
    if standalone_query == "NOT_RELEVANT" or not is_query_relevant(standalone_query):
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
    answer = generate_answer(query, context, images, language, is_comparison=is_comp)
    print(f"  >> Step 4 (LLM answer):   {time.time()-t0:.2f}s")
    
    print(f"  == TOTAL: {time.time()-t_total:.2f}s")
    
    return {
        "answer": answer,
        "images": images,
        "sources": sources
    }
