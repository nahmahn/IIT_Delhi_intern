# Chatbots

Two retrieval-augmented generation (RAG) projects built for the textile heritage work. Both use **Pinecone** for vector retrieval and **Groq** for generation, and each reads `PINECONE_API_KEY` and `GROQ_API_KEY` from its own local `.env` file (gitignored, never committed).

| Folder | What it is |
|---|---|
| [website_chatbot/](website_chatbot/) | FastAPI chatbot for the Textile Dept website, answering questions from the department's heritage documentation PDFs. Deployed as a Docker-based Hugging Face Space (configured by the front-matter in the repository root README) |
| [ask_textile/](ask_textile/) | Full-stack "Ask Textile" platform over textile course content: React frontend, Node/Prisma middleware, and a Python RAG service with a RAGAS evaluation harness |
