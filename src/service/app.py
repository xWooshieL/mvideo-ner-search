"""FastAPI service for query entity extraction + demo UI."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.service.extractor import QueryEntityExtractor

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"


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
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def ui() -> FileResponse:
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found (web/index.html)")
    return FileResponse(index)


@app.get("/health")
def health() -> Dict[str, Any]:
    ready = False
    detail = "ok"
    try:
        get_extractor()
        ready = True
    except Exception as e:
        detail = str(e)
    return {"status": "ok" if ready else "degraded", "extractor_ready": ready, "detail": detail}


@app.get("/metrics/summary")
def metrics_summary() -> Dict[str, Any]:
    path = ROOT / "artifacts" / "metrics.json"
    if not path.exists():
        return {"available": False}
    return {"available": True, "metrics": json.loads(path.read_text(encoding="utf-8"))}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> Dict[str, Any]:
    try:
        extractor = get_extractor()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Extractor not ready: {e}") from e
    return extractor.extract(req.query)


@app.post("/extract/debug")
def extract_debug(req: ExtractRequest) -> Dict[str, Any]:
    """Full extract + BIO pipelines for the lab UI."""
    try:
        extractor = get_extractor()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Extractor not ready: {e}") from e
    return extractor.extract_debug(req.query)


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
