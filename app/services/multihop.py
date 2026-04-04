from langchain_openai import ChatOpenAI
from app.core.config import OPENAI_API_KEY

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY
)

def decompose_question(question: str):
    prompt = f"""
Break the question into 2-3 smaller sub-questions.

Rules:
- Keep them short
- Make them searchable
- Focus on factual parts

Question:
{question}

Return only bullet points.
"""

    response = llm.invoke(prompt)

    lines = response.content.split("\n")

    sub_questions = [
        q.strip("- ").strip()
        for q in lines
        if q.strip()
    ]

    return sub_questions[:3]