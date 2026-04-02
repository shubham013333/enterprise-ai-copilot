from fastapi import APIRouter, UploadFile
import os
from app.services.rag import process_pdf

router = APIRouter()

UPLOAD_DIR = "data"

@router.post("/upload")
async def upload(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    process_pdf(file_path)

    return {"message" :f"{file.filename} processed successfully."}

