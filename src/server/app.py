"""Лёгкий FastAPI: query → факты каскада (SpellFix → rules → CRF → brand/attr).

Запуск из корня репозитория:

    fastapi run src/server/app.py --port 8000
    # или
    uvicorn src.server.app:app --host 0.0.0.0 --port 8000

Эндпоинты:
    GET  /health
    GET  /extract?query=...
    POST /extract   {"query": "..."}
    GET|POST /extract/debug  — то же + BIO rules/CRF + spell_fixes
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class ExtractRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Поисковый запрос")


class EntityOut(BaseModel):
    text: str
    label: str
    span: Optional[list[int]] = None
    attr_type: Optional[str] = None


class FactsResponse(BaseModel):
    query: str
    brand: Optional[str] = None
    category: Optional[str] = None
    model: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    entities: list[EntityOut] = Field(default_factory=list)
    latency_ms: float = 0.0
    spell_fixes: list[dict[str, Any]] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_extractor():
    from src.service.extractor import QueryEntityExtractor

    artifacts = Path(os.environ.get("MVIDEO_ARTIFACTS", ROOT / "artifacts"))
    models = Path(os.environ.get("MVIDEO_MODELS", ROOT / "models"))
    return QueryEntityExtractor.from_artifacts(artifacts, models)


def _to_facts(raw: dict[str, Any]) -> dict[str, Any]:
    ents = []
    for e in raw.get("entities") or []:
        ents.append(
            {
                "text": e.get("text"),
                "label": e.get("label"),
                "span": e.get("span"),
                "attr_type": e.get("attr_type"),
            }
        )
    return {
        "query": raw.get("query"),
        "brand": raw.get("brand"),
        "category": raw.get("category"),
        "model": raw.get("model"),
        "attributes": raw.get("attributes") or {},
        "entities": ents,
        "latency_ms": raw.get("latency_ms") or 0.0,
        "spell_fixes": raw.get("spell_fixes") or [],
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_extractor()
    yield


app = FastAPI(
    title="M.Video NER Facts API",
    description="MVP: извлечение фактов из поискового запроса (полный Python-каскад)",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        ex = get_extractor()
        return {
            "status": "ok",
            "crf": ex.ner_model is not None,
            "brand_clf": ex.brand_classifier is not None,
            "attr_clf": ex.use_attr_clf,
            "spellfix": ex.spell_fixer is not None,
            "category_clf": False,  # намеренно отключён
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/extract", response_model=FactsResponse)
def extract_get(query: str = Query(..., min_length=1)) -> dict[str, Any]:
    return _extract(query, debug=False)


@app.post("/extract", response_model=FactsResponse)
def extract_post(req: ExtractRequest) -> dict[str, Any]:
    return _extract(req.query, debug=False)


@app.get("/extract/debug")
def extract_debug_get(query: str = Query(..., min_length=1)) -> dict[str, Any]:
    return _extract(query, debug=True)


@app.post("/extract/debug")
def extract_debug_post(req: ExtractRequest) -> dict[str, Any]:
    return _extract(req.query, debug=True)


def _extract(query: str, *, debug: bool) -> dict[str, Any]:
    try:
        ex = get_extractor()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Extractor not ready: {e}") from e
    raw = ex.extract_debug(query) if debug else ex.extract(query)
    if debug:
        return raw
    return _to_facts(raw)
