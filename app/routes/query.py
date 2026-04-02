from fastapi import APIRouter 

router = APIRouter()

@router.post("/query")
def query(data: dict):
    question = data.get("question")

    # TODO: Connect RAG here

    return {
        "question" : question,
        "answer" : "RAG not implemented yet"
    }