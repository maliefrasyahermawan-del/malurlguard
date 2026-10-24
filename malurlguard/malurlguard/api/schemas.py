"""Pydantic request/response schemas for the MalUrlGuard API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator, HttpUrl


class PredictRequest(BaseModel):
    url: str
    explain: bool = False

    @field_validator("url")
    @classmethod
    def url_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("url must not be empty")
        if len(v) > 2048:
            raise ValueError("url must not exceed 2048 characters")
        return v


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: Optional[float] = None
    direction: str


class PredictResponse(BaseModel):
    url: str
    label: str
    label_id: int
    confidence: float
    probabilities: Dict[str, float]
    top_features: Optional[List[FeatureContribution]] = None
    latency_ms: float


class BatchPredictRequest(BaseModel):
    urls: List[str]
    explain: bool = False

    @field_validator("urls")
    @classmethod
    def urls_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("urls list must not be empty")
        if len(v) > 500:
            raise ValueError("Maximum batch size is 500 URLs")
        return [u.strip() for u in v if u.strip()]


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool


class MetricsResponse(BaseModel):
    models: Dict[str, Any]
    feature_count: int
