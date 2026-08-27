# from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# from retreiver import get_retriver
# from llm import get_llm
# from prompts import get_prompts

# main_retriever = get_retriver()
# llm = get_llm()
# prompt = get_prompts()

# def context(docs):
#   return "\n\n".join(doc.page_content for doc in docs)

# context_chain = RunnableParallel({
#     "context" : main_retriever | RunnableLambda(context),
#     "question" : RunnablePassthrough()

# })

# parser = StrOutputParser()

# main_chain = context_chain | prompt | llm | parser

# answer = main_chain.invoke("tell me about scope of textile")
# print(answer)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from retreiver import get_retriver
from llm import get_llm
from prompts import get_prompts

# --- FastAPI app ---
app = FastAPI(title="Textile RAG API")

# --- Load RAG components once at startup (not on every request) ---
print("Loading RAG pipeline...")
main_retriever = get_retriver()
llm = get_llm()
prompt = get_prompts()

def context(docs):
    return "\n\n".join(doc.page_content for doc in docs)

context_chain = RunnableParallel({
    "context": main_retriever | RunnableLambda(context),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = context_chain | prompt | llm | parser
print("RAG pipeline ready!")

# --- Request and Response models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

# --- Health check endpoint ---
@app.get("/health")
def health():
    return {"status": "ok"}

# --- Main RAG endpoint ---
@app.post("/rag", response_model=QueryResponse)
def ask(request: QueryRequest):
    try:
        answer = main_chain.invoke(request.question)
        return QueryResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))