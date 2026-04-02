from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.services.embeddings import get_embeddings
from app.db.vector_store import save_vector_store
from app.services.bm25 import build_bm25

def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = file_path

    build_bm25(chunks)


    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)

    save_vector_store(db)