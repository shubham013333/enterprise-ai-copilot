from app.routes.query import query as rag_query

def run_rag(question: str):
    result = rag_query({"question": question})
    return result["answer"]