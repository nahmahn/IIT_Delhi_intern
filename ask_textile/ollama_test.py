from langchain_community.chat_models import init_chat_model

llm = init_chat_model(
    model_name="qwen2.5:7b",
    temperature=0,
    # other params...
)

messages = "tell me about lucknow" 
ai_msg = llm.invoke(messages)
print(ai_msg.content)