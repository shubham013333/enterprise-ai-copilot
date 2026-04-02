from fastapi import APIRouter, UploadFile

router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile):
    content = await file.read()

    # TODO: Process PDF -> Emdeddings

    return {"filename" :file.filename}



