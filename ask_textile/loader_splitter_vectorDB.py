# import langchain

# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.document_loaders import UnstructuredPDFLoader

# loader = DirectoryLoader(
#     "chatbot/downloads",
#     glob="**/*.pdf",
#     loader_cls=UnstructuredPDFLoader,
#     use_multithreading=True,
#     show_progress=True
# )

# documents = loader.load()

from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Loader
loader = DirectoryLoader(
    "chatbot/downloads",
    glob="**/*.pdf",
    loader_cls=UnstructuredPDFLoader,
    use_multithreading=True,
    show_progress=True
)

# 2. Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 3. Embeddings
# from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 4. Vector DB
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_qwen_db"
)

# 5. Ingestion
for doc in loader.lazy_load():
    chunks = text_splitter.split_documents([doc])
    vectorstore.add_documents(chunks)

vectorstore.persist()

print("RAG ingestion complete.")