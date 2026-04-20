from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from .db import Base, RagQueryLog, PredictionLog, create_engine, create_session_factory
from .model_inference import load_model, predict_image_bytes
from .rag import SimpleRagIndex
from .schemas import HealthResponse, PredictResponse, RagAskRequest, RagAskResponse, RagChunk


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine = create_engine()
    session_factory = create_session_factory(engine)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device, checkpoint_path=os.getenv("SELF_PRUNING_CKPT"))

    docs_dir = os.getenv("RAG_DOCS_DIR", os.path.join(os.path.dirname(__file__), "..", "rag_docs"))
    rag = SimpleRagIndex(docs_dir=os.path.abspath(docs_dir))

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.device = device
    app.state.model = model
    app.state.rag = rag
    yield

    await engine.dispose()


app = FastAPI(title="Self-Pruning NN API (FastAPI + RAG + SQL)", version="1.0.0", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def dashboard_index():
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"error": "Dashboard not found. Missing api/static/index.html"}
    return FileResponse(str(index_path))


def _default_sweep_results_path() -> Path:
    # Prefer a full run, fallback to smoke.
    env = os.getenv("SWEEP_RESULTS_PATH")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent.parent
    p1 = here / "outputs" / "lambda_sweep_results.json"
    p2 = here / "outputs_smoke" / "lambda_sweep_results.json"
    return p1 if p1.exists() else p2


@app.get("/dashboard/sweep")
async def dashboard_sweep():
    path = _default_sweep_results_path()
    if not path.exists():
        return {"error": f"Sweep results not found at {str(path)}"}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/dashboard/info")
async def dashboard_info():
    return {
        "device": str(app.state.device),
        "checkpoint": os.getenv("SELF_PRUNING_CKPT"),
        "docs_dir": os.getenv("RAG_DOCS_DIR"),
        "sweep_results_path": str(_default_sweep_results_path()),
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    img_bytes = await file.read()
    pred, conf = predict_image_bytes(app.state.model, app.state.device, img_bytes)

    async with app.state.session_factory() as session:
        session.add(PredictionLog(filename=file.filename, predicted_class=pred, confidence=conf))
        await session.commit()

    return PredictResponse(predicted_class=pred, confidence=conf)


@app.post("/rag/ask", response_model=RagAskResponse)
async def rag_ask(req: RagAskRequest) -> RagAskResponse:
    answer, model_name, retrieved = await app.state.rag.answer(
        req.question, top_k=req.top_k, use_llm=req.use_llm
    )

    async with app.state.session_factory() as session:
        session.add(RagQueryLog(question=req.question, top_k=req.top_k, answer=answer, model=model_name))
        await session.commit()

    return RagAskResponse(
        answer=answer,
        model=model_name,
        retrieved=[
            RagChunk(source=d.source, text=d.text, score=score) for d, score in retrieved
        ],
    )


@app.get("/logs/rag/recent")
async def recent_rag_logs(limit: int = 20):
    limit = max(1, min(200, limit))
    async with app.state.session_factory() as session:
        rows = (await session.execute(select(RagQueryLog).order_by(RagQueryLog.id.desc()).limit(limit))).scalars().all()
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "question": r.question,
            "top_k": r.top_k,
            "model": r.model,
        }
        for r in rows
    ]


@app.get("/logs/predictions/recent")
async def recent_prediction_logs(limit: int = 20):
    limit = max(1, min(200, limit))
    async with app.state.session_factory() as session:
        rows = (await session.execute(select(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit))).scalars().all()
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "filename": r.filename,
            "predicted_class": r.predicted_class,
            "confidence": r.confidence,
        }
        for r in rows
    ]

