from langchain_openai import ChatOpenAI
from app.core.config import OPENAI_API_KEY

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY
)

def route_query(question: str):
    prompt = f"""
Classify the user query into ONE category:

- rag (needs document retrieval)
- math (needs calculation)
- general (simple answer)

Query:
{question}

Return only one word.
"""

    response = llm.invoke(prompt)

    return response.content.strip().lower()