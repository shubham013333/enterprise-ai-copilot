from fastapi import APIRouter
from app.db.vector_store import get_vector_store
from langchain_openai import ChatOpenAI
from app.services.chat_memory import add_to_memory, get_memory

router = APIRouter()

@router.post("/query")
def query(data: dict):
    question = data.get("question")

    db = get_vector_store()

    if not db:
        return {"error": "No documents uploaded yet"}

    docs = db.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    memory = get_memory()
    history_text = "\n".join([
        f"User: {m.get('user', '')}\nAI: {m.get('ai', '')}"
        for m in memory
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = f"""
You are a professional AI assistant.

Rules:
- Answer ONLY from context
- If not in context → say "I don't know"
- Be concise and clear
- Use bullet points if needed

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