from langchain_openai import ChatOpenAI
from app.core.config import OPENAI_API_KEY

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY
)


def decompose_question(question: str):
    prompt = f"""
Break the question into 2-3 smaller factual sub-questions.

Question:
{question}

Return bullet points only.
"""

    response = llm.invoke(prompt)

    lines = response.content.split("\n")

    return [
        q.strip("- ").strip()
        for q in lines
        if q.strip()
    ][:3]