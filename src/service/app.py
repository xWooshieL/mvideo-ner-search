"""FastAPI service for query entity extraction."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.service.extractor import QueryEntityExtractor

ROOT = Path(__file__).resolve().parents[2]


class ExtractRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Поисковый запрос пользователя")


class ExtractResponse(BaseModel):
    query: str
    entities: list
    brand: Optional[str] = None
    category: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float


@lru_cache(maxsize=1)
def get_extractor() -> QueryEntityExtractor:
    artifacts = Path(os.environ.get("MVIDEO_ARTIFACTS", ROOT / "artifacts"))
    models = Path(os.environ.get("MVIDEO_MODELS", ROOT / "models"))
    return QueryEntityExtractor.from_artifacts(artifacts, models)


app = FastAPI(
    title="M.Video Intelligent Search NER",
    description="Извлечение сущностей (BRAND, CATEGORY, ATTR) из поисковых запросов",
    version="1.0.0",
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> Dict[str, Any]:
    try:
        extractor = get_extractor()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Extractor not ready: {e}") from e
    return extractor.extract(req.query)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.service.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
