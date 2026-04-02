from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.services.embeddings import get_embeddings
from app.db.vector_store import save_vector_store

def process_pdf(file_path: str):
    loader  = PyPDFLoader(file_path)
    documents = loader.load()   

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)

    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    db = FAISS.from_documents(chunks, embeddings)

    save_vector_store(db)

    
