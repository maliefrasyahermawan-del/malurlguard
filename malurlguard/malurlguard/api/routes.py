"""FastAPI route handlers."""

import logging
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from malurlguard import __version__
from malurlguard.inference.explainer import ShapExplainer
from malurlguard.inference.predictor import Predictor
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    FeatureContribution,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level singletons (lazy-loaded)
_predictor: Predictor | None = None
_explainer: ShapExplainer | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def get_explainer() -> ShapExplainer:
    global _explainer
    if _explainer is None:
        _explainer = ShapExplainer()
    return _explainer


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    try:
        predictor = get_predictor()
        predictor._load_models()
        loaded = predictor._loaded
    except Exception:
        loaded = False
    return HealthResponse(status="ok", version=__version__, models_loaded=loaded)


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """Classify a single URL."""
    try:
        predictor = get_predictor()
        result = predictor.predict(req.url)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed for %s", req.url)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    top_features = None
    if req.explain:
        try:
            explainer = get_explainer()
            raw_features = explainer.explain(req.url)
            top_features = [FeatureContribution(**f) for f in raw_features]
        except Exception as exc:
            logger.warning("Explanation failed: %s", exc)

    return PredictResponse(
        url=req.url,
        top_features=top_features,
        **result,
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Classify multiple URLs in one call (max 500)."""
    t0 = time.perf_counter()
    try:
        predictor = get_predictor()
        results_raw = predictor.predict_batch(req.urls)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}")

    total_latency = round((time.perf_counter() - t0) * 1000, 3)

    responses = []
    for r in results_raw:
        top_features = None
        if req.explain:
            try:
                explainer = get_explainer()
                raw = explainer.explain(r["url"])
                top_features = [FeatureContribution(**f) for f in raw]
            except Exception:
                pass
        responses.append(PredictResponse(top_features=top_features, **r))

    return BatchPredictResponse(
        results=responses,
        total=len(responses),
        total_latency_ms=total_latency,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Return model info and feature count."""
    try:
        predictor = get_predictor()
        model_info = predictor.model_info
        feat_count = len(predictor._feature_names)
    except Exception:
        model_info = {}
        feat_count = 0
    return MetricsResponse(models=model_info, feature_count=feat_count)
