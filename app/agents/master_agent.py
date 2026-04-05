from app.agents.router import route_query
from app.agents.rag_agent import run_rag
from app.agents.tools import calculator_tool
from langchain_openai import ChatOpenAI
from app.core.config import OPENAI_API_KEY

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY
)

def run_agent(question: str):
    route = route_query(question)

    if "math" in route:
        return calculator_tool(question)

    elif "rag" in route:
        return run_rag(question)

    else:
        response = llm.invoke(question)
        return response.content