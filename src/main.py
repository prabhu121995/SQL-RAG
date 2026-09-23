import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.database import get_sql_database
from src.models import AskRequest, AskResponse, HealthResponse
from src.rag import SQLRAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

service: SQLRAGService | None = None
db_tables: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the DB and build the RAG service once at startup."""
    global service, db_tables
    logger.info("Starting up SQL RAG API...")
    db = get_sql_database()
    db_tables = db.get_usable_table_names()
    service = SQLRAGService(db)
    logger.info("Ready. Tables: %s", db_tables)
    yield
    logger.info("Shutting down SQL RAG API.")


app = FastAPI(
    title="SQL RAG API",
    description="Natural-language Q&A over a telecom support tickets SQLite DB.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", tables=db_tables)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        out = service.ask(payload.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Internal error") from e

    return AskResponse(
        question=payload.question,
        sql=out["sql"],
        result=out["result"],
        answer=out["answer"],
    )