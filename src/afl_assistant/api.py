"""FastAPI service for the AFL assistant."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .graph import run_agent
from .predict import predict_match_winner, predict_top_player

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AFL Match Assistant API",
    description="Domain-locked AFL chat, retrieval and prediction service.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    intent: str
    needs_clarification: bool = False


class PredictionRequest(BaseModel):
    team_a: str = Field(..., examples=["Collingwood Magpies"])
    team_b: str = Field(..., examples=["Carlton Blues"])


class PredictionResponse(BaseModel):
    team_a: str
    team_b: str
    winner: str
    probability: float
    confidence: str
    disclaimer: str


@app.get("/")
async def root() -> Dict:
    return {"name": "AFL Match Assistant API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health() -> Dict:
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = run_agent(request.message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(**result)


@app.post("/predict/match", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest) -> PredictionResponse:
    try:
        result = predict_match_winner(request.team_a, request.team_b)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictionResponse(
        team_a=result["team_a"],
        team_b=result["team_b"],
        winner=result["winner"],
        probability=result["probability"],
        confidence=result["confidence"],
        disclaimer="This is a prediction based on historical data, not a certainty.",
    )


@app.get("/predict/player")
async def predict_player(team: Optional[str] = None, stat_type: str = "disposals", top_k: int = 5):
    try:
        return {"players": predict_top_player(team=team, stat_type=stat_type, top_k=top_k)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
