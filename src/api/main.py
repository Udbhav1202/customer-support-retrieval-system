
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from src.rag.rag_service import RAGService

app = FastAPI(
    title="Verizon Support Agent",
    description="AI-powered customer support using hybrid retrieval and RAG.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService()


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceCase(BaseModel):
    rank: int
    customer_message: str
    support_response: str
    rrf_score: float



class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceCase]



@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=422,
            detail="Query must not be empty.",
        )

    try:
        result = rag_service.answer_with_sources(
            customer_query=request.query,
            top_k=request.top_k,
        )

        return ChatResponse(
            query=request.query,
            answer=result["answer"],
            sources=result["sources"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The support agent could not generate a response.",
        ) from exc
