import os
import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from model import GraphRAGPipeline

# -------------------------------------------------------------------
# Bootstrap
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GraphRAG-Gateway")

load_dotenv("content.env")
load_dotenv("neo4j.env")


# -------------------------------------------------------------------
# Required configuration (explicit + validated)
# -------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("openai_api_key")
AZURE_OPENAI_ENDPOINT = os.getenv("azure_endpoint")
OPENAI_API_VERSION = os.getenv("openai_api_version")
AZURE_OPENAI_DEPLOYMENT = os.getenv("deployment_name")

NEO4J_URI = os.getenv("uri")
NEO4J_USERNAME = os.getenv("username")
NEO4J_PASSWORD = os.getenv("password")

missing = [
    name for name, value in {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "OPENAI_API_VERSION": OPENAI_API_VERSION,
        "AZURE_OPENAI_DEPLOYMENT": AZURE_OPENAI_DEPLOYMENT,
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
    }.items()
    if not value
]

if missing:
    raise RuntimeError(f"Missing required env vars: {missing}")

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(title="GraphRAG Gateway", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[GraphRAGPipeline] = None

# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

# -------------------------------------------------------------------
# Startup / Shutdown
# -------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global pipeline

    logger.info("Initializing GraphRAGPipeline...")

    pipeline = GraphRAGPipeline(
        openai_api_key=OPENAI_API_KEY,
        openai_api_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        neo4j_uri=NEO4J_URI,
        neo4j_username=NEO4J_USERNAME,
        neo4j_password=NEO4J_PASSWORD,
    )

    logger.info("GraphRAGPipeline ready.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down GraphRAG Gateway.")

# -------------------------------------------------------------------
# Internal helper
# -------------------------------------------------------------------
async def _invoke_pipeline(question: str):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        return await pipeline.run(question)
    except Exception as e:
        logger.exception("Pipeline failure: %s", e)
        raise HTTPException(status_code=500, detail="Pipeline execution failed")

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"ok": pipeline is not None}


@app.post("/query")
async def query(req: QueryRequest):
    start = time.time()

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    if req.top_k is not None:
        pipeline.top_k = int(req.top_k)

    state = await _invoke_pipeline(question)

    return {
        "response": state.get("final_answer", ""),
        "metadata": {
            "decision": state.get("decision"),
            "cypher_query": state.get("cypher_query"),
            "results": state.get("results", []),
            "runtime_seconds": round(time.time() - start, 3),
        },
    }


@app.post("/cypher_preview")
async def cypher_preview(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    state = await _invoke_pipeline(question)

    return {
        "cypher_query": state.get("cypher_query", ""),
        "decision": state.get("decision"),
    }
