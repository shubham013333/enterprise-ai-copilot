from fastapi import FastAPI
from app.routes import query, upload

app = FastAPI(title="enterprise-ai-copilot")

app.include_router(query.router)
app.include_router(upload.router)

@app.get("/")
def root():
    return {"message":"welcome to enterprise-ai-copilot!"}

