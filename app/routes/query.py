from fastapi import APIRouter
from app.db.vector_store import get_vector_store
from langchain_openai import ChatOpenAI

router = APIRouter()

@router.post("/query")
def query(data: dict):
    question = data.get("question")

    db = get_vector_store()

    if not db:
        return {"error": "No documents uploaded yet"}

    docs = db.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI()

    response = llm.invoke(
        f"Answer based on context:\n{context}\n\nQuestion: {question}"
    )

    return {
        "answer": response.content,
        "sources": [doc.page_content[:200] for doc in docs]
    }