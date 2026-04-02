from langchain_community.vectorstores import FAISS

VECTOR_DB = None

def save_vector_store(vector_store):
    global VECTOR_DB
    if VECTOR_DB is None:
        VECTOR_DB = vector_store
    else:
        VECTOR_DB.merge_from(vector_store)

def get_vector_store():
    return VECTOR_DB