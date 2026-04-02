from fastapi import APIRouter
from app.db.vector_store import get_vector_store
from langchain_openai import ChatOpenAI
from app.services.chat_memory import add_to_memory, get_memory
from app.services.reranker import rerank


router = APIRouter()

@router.post("/query")
def query(data: dict):
    question = data.get("question")

    db = get_vector_store()

    if not db:
        return {"error": "No documents uploaded yet"}

    docs = db.similarity_search(question, k=15)

    docs = rerank(question, docs)[:5] #rerank and take top 3

    context = "\n".join([doc.page_content for doc in docs])

    memory = get_memory()
    history_text = "\n".join([
        f"User: {m.get('user', '')}\nAI: {m.get('ai', '')}"
        for m in memory
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = f"""
You are a strict AI assistant.

Rules:
- Answer ONLY using the provided context
- DO NOT say you don't have access
- DO NOT use outside knowledge
- If answer not found → say "Not found in document"
- Be precise and factual

Conversation history:
{history_text}

Context:
{context}

User Question:
{question}

Answer clearly and based on context.
"""

    response = llm.invoke(prompt)

    add_to_memory(question, response.content)

    return {
        "answer": response.content,
        "sources": [doc.page_content[:200] for doc in docs]
    }