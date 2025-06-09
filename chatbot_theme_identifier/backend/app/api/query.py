from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm_query import process_query

router = APIRouter(prefix="/query", tags=["Query"])

class QueryRequest(BaseModel):
    question: str

@router.post("/")
def ask_query(request: QueryRequest):
    return process_query(request.question)