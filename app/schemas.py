# app/schemas.py
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Features d'un salarié")


class PredictBatchRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="Liste de features par salarié")


class PredictResponse(BaseModel):
    proba: float = Field(..., ge=0.0, le=1.0)
    label: int = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0.0, le=1.0)
    risk_label: str = Field(..., description="Libellé de risque compréhensible")
