from fastapi import APIRouter
from app.db.vector_store import get_vector_store
from langchain_openai import ChatOpenAI
from app.services.chat_memory import add_to_memory, get_memory
from app.services.reranker import rerank
from app.services.bm25 import bm25_search

router = APIRouter()

@router.post("/query")
def query(data: dict):
    question = data.get("question")

    db = get_vector_store()

    if not db:
        return {"error": "No documents uploaded yet"}

    # Vector search
    vector_docs = db.similarity_search(question, k=10)

    #  BM25 search
    bm25_docs = bm25_search(question, k=5)

    # Combine results
    all_docs = vector_docs + bm25_docs

    # Remove duplicates (stable way)
    seen = set()
    unique_docs = []

    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    #  Rerank (limit input size for performance)
    docs = rerank(question, unique_docs[:15])[:5]

    # Build context from BEST docs
    context = "\n".join([doc.page_content for doc in docs])

    #  Memory
    memory = get_memory()
    history_text = "\n".join([
        f"User: {m.get('user', '')}\nAI: {m.get('ai', '')}"
        for m in memory
    ])

    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Strong RAG Prompt
    prompt = f"""
You are a strict AI assistant.

Rules:
- Answer ONLY using the provided context
- DO NOT use outside knowledge
- If answer not found → say "Not found in document"
- Be precise and factual
- Keep answers concise

Conversation history:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    # 💾 Save memory
    add_to_memory(question, response.content)

    return {
        "answer": response.content,
        "sources": [
            {
                "content": doc.page_content[:200],
                "source": doc.metadata.get("source", "unknown")
            }
            for doc in docs
        ]
    }