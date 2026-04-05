from langchain_openai import ChatOpenAI
from app.core.config import OPENAI_API_KEY

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY
)


def rewrite_query(query: str):
    prompt = f"""
Rewrite the query into 3 improved search queries.

- Make them specific
- Add keywords
- Keep them short

Query:
{query}

Return as bullet points.
"""

    response = llm.invoke(prompt)

    lines = response.content.split("\n")

    queries = [
        q.strip("- ").strip()
        for q in lines
        if q.strip()
    ]

    return list(set(queries))[:3]