"""
FastAPI Server for Voice-Enabled RAG Pipeline.
Provides REST and WebSocket endpoints for voice input, retrieval, structured generation,
and live latency telemetry.
"""

import os
import io
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.harness import StructuredModelHarness, RAGRequest, RAGResponse
from src.stt_service import SarvamSTTService
from src.config import BASE_DIR, DATA_DIR, EMBEDDING_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice-Enabled MSMARCO RAG Engine",
    description="Sub-200ms Voice-Enabled RAG with Multi-Strategy Chunking, Qdrant Vector Indexing, and Structured Guardrails",
    version="1.0.0"
)

# Enable CORS for local development and browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons for zero-cold-start initialization
_harness: Optional[StructuredModelHarness] = None
_stt_service: Optional[SarvamSTTService] = None


def get_harness() -> StructuredModelHarness:
    global _harness
    if _harness is None:
        logger.info("Lazy initializing StructuredModelHarness...")
        _harness = StructuredModelHarness()
    return _harness


def get_stt_service() -> SarvamSTTService:
    global _stt_service
    if _stt_service is None:
        logger.info("Lazy initializing SarvamSTTService...")
        _stt_service = SarvamSTTService()
    return _stt_service


@app.on_event("startup")
async def startup_event():
    logger.info("RAG Application starting up...")


@app.get("/api/health")
async def health_check():
    """Health check and status telemetry."""
    collections = []
    try:
        h = get_harness()
        cols = h.retriever.client.get_collections().collections
        collections = [c.name for c in cols]
    except Exception as e:
        logger.warning(f"Could not load collections during health check: {e}")
        collections = ["msmarco_semantic_chunks", "msmarco_passage_chunks", "msmarco_sentence_chunks", "msmarco_hierarchical_chunks"]

    return {
        "status": "online",
        "service": "Voice-Enabled RAG Engine",
        "active_collections": collections,
        "embedding_model": EMBEDDING_MODEL_NAME
    }


@app.post("/api/query", response_model=RAGResponse)
async def process_text_query(request: RAGRequest):
    """
    Direct text query endpoint executing guardrails, multi-strategy retrieval,
    reranking, and generation.
    """
    harness = get_harness()
    response = harness.run_pipeline(request, stt_latency_ms=0.0)
    return response


@app.post("/api/voice", response_model=RAGResponse)
async def process_voice_audio(
    audio: UploadFile = File(...),
    language: str = Form(default="hi-IN"),
    strategy: str = Form(default="all"),
    use_cross_encoder: bool = Form(default=False),
    top_k: int = Form(default=5)
):
    """
    Voice upload endpoint: transcribes audio via Sarvam AI saaras:v3,
    then executes full RAG pipeline with end-to-end latency profiling.
    """
    harness = get_harness()
    stt_service = get_stt_service()

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload received.")

    # 1. Transcribe voice audio
    transcript, stt_ms = stt_service.transcribe_audio_bytes(
        audio_bytes=audio_bytes,
        language_code=language,
        filename=audio.filename or "recording.wav"
    )

    # 2. Run structured RAG pipeline
    request = RAGRequest(
        query=transcript,
        language=language,
        strategy=strategy,
        use_cross_encoder=use_cross_encoder,
        top_k=top_k
    )
    response = harness.run_pipeline(request, stt_latency_ms=stt_ms)
    return response


@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time binary audio stream and low-latency response.
    """
    await websocket.accept()
    logger.info("WebSocket voice client connected.")
    harness = get_harness()
    stt_service = get_stt_service()
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                transcript, stt_ms = stt_service.transcribe_audio_bytes(
                    audio_bytes=audio_bytes,
                    language_code="hi-IN"
                )
                
                await websocket.send_json({
                    "type": "transcript",
                    "transcript": transcript,
                    "stt_ms": round(stt_ms, 2)
                })

                req = RAGRequest(query=transcript, top_k=5)
                res = harness.run_pipeline(req, stt_latency_ms=stt_ms)
                
                await websocket.send_json({
                    "type": "rag_response",
                    "data": res.model_dump()
                })

            elif "text" in message and message["text"]:
                import json
                try:
                    payload = json.loads(message["text"])
                    query_text = payload.get("query", "")
                    req = RAGRequest(
                        query=query_text, 
                        strategy=payload.get("strategy", "all"),
                        top_k=payload.get("top_k", 5)
                    )
                    res = harness.run_pipeline(req, stt_latency_ms=0.0)
                    await websocket.send_json({
                        "type": "rag_response",
                        "data": res.model_dump()
                    })
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket voice client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# Mount Static Frontend
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(frontend_dir / "index.html")
