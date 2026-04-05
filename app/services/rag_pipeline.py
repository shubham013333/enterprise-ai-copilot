from app.db.vector_store import get_vector_store
from langchain_openai import ChatOpenAI
from app.core.config import OPENAI_API_KEY
from app.services.chat_memory import add_to_memory, get_memory
from app.services.reranker import rerank
from app.services.bm25 import bm25_search
from app.services.query_rewriter import rewrite_query
from app.services.multihop import decompose_question

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY
)


def run_rag_pipeline(question: str):
    db = get_vector_store()

    if not db:
        return {
            "answer": "No documents uploaded yet",
            "sources": []
        }

    # Multi-hop (controlled)
    if len(question.split()) < 6:
        sub_questions = [question]
    else:
        sub_questions = decompose_question(question)
        sub_questions.append(question)

    all_docs = []

    # Query rewriting + hybrid search
    for sq in sub_questions:
        queries = rewrite_query(sq)
        queries.append(sq)

        for q in queries:
            try:
                vector_docs = db.similarity_search(q, k=4)
                bm25_docs = bm25_search(q, k=2)
                all_docs.extend(vector_docs + bm25_docs)
            except Exception:
                continue

    # Deduplication
    seen = set()
    unique_docs = []

    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    # Rerank
    docs = rerank(question, unique_docs[:20])[:5]

    # Context
    context = "\n".join([doc.page_content for doc in docs])

    # Memory
    memory = get_memory()
    history_text = "\n".join([
        f"User: {m.get('user', '')}\nAI: {m.get('ai', '')}"
        for m in memory
    ])

    prompt = f"""
You are a strict AI assistant.

Rules:
- Answer ONLY using the provided context
- DO NOT use outside knowledge
- If answer not found → say "Not found in document"
- Be precise and concise

Conversation history:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content

    add_to_memory(question, answer)

    return {
        "answer": answer,
        "sources": [
            {
                "content": doc.page_content[:200],
                "source": doc.metadata.get("source", "unknown")
            }
            for doc in docs
        ]
    }