from fastapi import APIRouter
from langchain_openai import ChatOpenAI
from app.core.config import OPENAI_API_KEY
from app.services.rag_pipeline import run_rag_pipeline

router = APIRouter()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY
)


def is_math_query(q: str):
    return any(char.isdigit() for char in q)


@router.post("/query")
def query(data: dict):
    question = data.get("question")

    # 1. Math handling
    if is_math_query(question):
        try:
            return {"answer": str(eval(question))}
        except Exception:
            pass

    # 2. RAG
    rag_result = run_rag_pipeline(question)

    if rag_result["answer"].lower().startswith("not found"):
        # 3. Fallback LLM
        response = llm.invoke(question)
        return {
            "answer": response.content,
            "sources": []
        }

    return rag_result