from langchain_community.chat_models import ChatOllama

def get_llm():
    return ChatOllama(
        model="gemma3:4b",
        temperature=0.3,
    )