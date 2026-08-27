from langchain_core.prompts import PromptTemplate

def get_prompts():
    prompt  = PromptTemplate(
    template = """you are textile expert 
    Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}""",
      input_variables= ["context","question"]
    )
    return prompt