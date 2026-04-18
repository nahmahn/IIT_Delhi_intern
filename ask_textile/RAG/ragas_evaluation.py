"""
RAGAS Evaluation Pipeline
--------------------------
Plugs directly into your existing RAG chain.

- Ground Truth Generation : GPT-4 Mini
- RAG                     : Your existing chain (qwen2.5:7b + ChromaDB)
- Judge LLM               : GPT-4o  (gpt-5 blocks temperature != 1, breaks RAGAS)
- Metrics                 : Faithfulness, Answer Relevancy,
                            Context Recall, Context Precision
"""

import os
import random
import json
import pandas as pd
from typing import List, Dict

# ── OpenAI (GPT-4 Mini for ground truth generation) ───────────────────────────
from openai import OpenAI

# ── Your existing RAG components ──────────────────────────────────────────────
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
# ── RAGAS ─────────────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

# ── Your modules (must be in same directory) ──────────────────────────────────
from retreiver import get_retriver
from llm import get_llm
from prompts import get_prompts

# =============================================================================
# CONFIGURATION  — only edit this block
# =============================================================================

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
GENERATION_MODEL = "gpt-4o-mini"   # cheap: generates ground truth
JUDGE_MODEL      = "gpt-4o"        # gpt-5 blocks temperature != 1, breaks RAGAS internals

NUM_QUESTIONS    = 30
RESULTS_PATH     = "ragas_results.csv"

# =============================================================================
# 1.  BUILD RAG — reuses your exact modules
# =============================================================================

def build_rag():
    """
    Reuses your get_retriver(), get_llm(), get_prompts().
    Returns:
        answer_chain : same chain as your main.py
        retriever    : kept separately to extract contexts for RAGAS
    """
    retriever = get_retriver()
    llm       = get_llm()
    prompt    = get_prompts()

    def context(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    context_chain = RunnableParallel({
        "context":  retriever | RunnableLambda(context),
        "question": RunnablePassthrough(),
    })

    answer_chain = context_chain | prompt | llm | StrOutputParser()

    return answer_chain, retriever


# =============================================================================
# 2.  SAMPLE CHUNKS FROM YOUR CHROMADB
# =============================================================================

def sample_chunks(n: int) -> List[str]:
    """
    Connects to your ChromaDB (same path as retreiver.py)
    and randomly samples n chunks.
    """
    print(f"\n[1/4] Sampling {n} chunks from ChromaDB ...")

    embeddings  = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_qwen_db",  # same as your retreiver.py
    )

    raw      = vectorstore._collection.get(include=["documents"])
    all_docs = raw["documents"]

    if len(all_docs) < n:
        print(f"  ⚠  Only {len(all_docs)} chunks available. Using all.")
        n = len(all_docs)

    sampled = random.sample(all_docs, n)
    print(f"  ✓  Sampled {len(sampled)} / {len(all_docs)} total chunks.")
    return sampled


# =============================================================================
# 3.  GENERATE Q&A PAIRS WITH GPT-4 MINI
# =============================================================================

def generate_qa_pairs(chunks: List[str]) -> List[Dict]:
    """
    Sends each chunk to GPT-4 Mini.
    Returns list of {question, ground_truth, source_chunk}.
    """
    print(f"\n[2/4] Generating Q&A pairs with {GENERATION_MODEL} ...")

    client   = OpenAI(api_key=OPENAI_API_KEY)
    qa_pairs = []

    for i, chunk in enumerate(chunks):
        prompt_text = f"""You are a question generation assistant for a textile domain RAG system.

Given the following text chunk, generate ONE realistic user question
that can be answered from this chunk, and the ground truth answer based
STRICTLY on the chunk. Do not use outside knowledge.

Text chunk:
\"\"\"
{chunk}
\"\"\"

Respond ONLY with valid JSON, no extra text:
{{
  "question": "...",
  "ground_truth": "..."
}}"""

        try:
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            pair                 = json.loads(response.choices[0].message.content)
            pair["source_chunk"] = chunk
            qa_pairs.append(pair)
            print(f"  ✓  [{i+1}/{len(chunks)}] {pair['question'][:80]}...")

        except Exception as e:
            print(f"  ✗  [{i+1}/{len(chunks)}] Skipped — {e}")
            continue

    print(f"  → {len(qa_pairs)} Q&A pairs generated.")
    return qa_pairs


# =============================================================================
# 4.  RUN YOUR RAG CHAIN ON EACH QUESTION
# =============================================================================

def run_rag_on_questions(
    answer_chain,
    retriever,
    qa_pairs: List[Dict],
) -> List[Dict]:
    """
    For each question:
      - answer_chain.invoke()  → answer string (your exact chain)
      - retriever.invoke()     → source docs   → contexts list for RAGAS
    """
    print(f"\n[3/4] Running RAG on {len(qa_pairs)} questions ...")

    results = []

    for i, pair in enumerate(qa_pairs):
        question = pair["question"]
        try:
            # Your chain returns a plain string (StrOutputParser)
            answer: str       = answer_chain.invoke(question)

            # Retriever returns List[Document] — extract text for RAGAS
            source_docs       = retriever.invoke(question)
            contexts: List[str] = [doc.page_content for doc in source_docs]

            results.append({
                "question":     question,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": pair["ground_truth"],
            })
            print(f"  ✓  [{i+1}/{len(qa_pairs)}] answered.")

        except Exception as e:
            print(f"  ✗  [{i+1}/{len(qa_pairs)}] RAG error — {e}")
            continue

    print(f"  → {len(results)} RAG responses collected.")
    return results


# =============================================================================
# 5.  RAGAS EVALUATION  (GPT-5 as judge)
# =============================================================================

def run_ragas_evaluation(results: List[Dict]) -> pd.DataFrame:
    """Build RAGAS dataset and run all 4 metrics with GPT-4o as judge."""

    print(f"\n[4/4] Evaluating with RAGAS — judge: {JUDGE_MODEL} ...")

    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    # Old-style classes from ragas/metrics/_*.py — these work with evaluate()
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model=JUDGE_MODEL, api_key=OPENAI_API_KEY)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    )

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextRecall(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
    ]

    eval_result = evaluate(dataset=dataset, metrics=metrics)
    return eval_result.to_pandas()

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  RAGAS Evaluation — Textile RAG (qwen2.5:7b + ChromaDB)")
    print("=" * 60)

    answer_chain, retriever = build_rag()
    chunks                  = sample_chunks(NUM_QUESTIONS)
    qa_pairs                = generate_qa_pairs(chunks)
    results                 = run_rag_on_questions(answer_chain, retriever, qa_pairs)

    if not results:
        print("No results collected. Exiting.")
        return

    scores_df = run_ragas_evaluation(results)
    scores_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n✓ Per-question results saved → {RESULTS_PATH}")

    print("\n" + "=" * 60)
    print("  AGGREGATE SCORES  (0 = worst, 1 = best)")
    print("=" * 60)
    for col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        if col in scores_df.columns:
            print(f"  {col:<25} {scores_df[col].mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()