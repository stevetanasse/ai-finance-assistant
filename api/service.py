"""FastAPI service wrapping the LangGraph finance assistant.

Run with:
    uv run uvicorn api.service:app --host 0.0.0.0 --port 8000

Override the Qdrant collection at runtime via env var:
    COLLECTION_NAME=fin_c500_o50_bge-small_bm42 uv run uvicorn api.service:app --port 8000
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.workflow.graph import build_graph

load_dotenv()

_graph = None  # built once at startup, shared across requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    llm = ChatOpenAI(model="gpt-4o-mini")
    # COLLECTION_NAME absent → build_graph derives name from config.yaml
    collection_name = os.environ.get("COLLECTION_NAME")
    _graph = build_graph(llm, collection_name=collection_name)
    yield
    _graph = None


app = FastAPI(title="AI Finance Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    response: str
    route: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        config = {"configurable": {"thread_id": req.thread_id}}
        result = _graph.invoke(
            {"messages": [HumanMessage(content=req.message)], "call_counts": {}},
            config=config,
        )
        return ChatResponse(
            response=result["messages"][-1].content,
            route=result["route"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
