from rank_bm25 import BM25Okapi

bm25 = None
documents_store = []

def build_bm25(new_docs):
    global bm25, documents_store

    documents_store.extend(new_docs)
    tokenized_docs = [doc.page_content.split() for doc in documents_store]

    bm25 = BM25Okapi(tokenized_docs)


def bm25_search(query, k=5):
    if not bm25:
        return []

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(documents_store, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:k]]