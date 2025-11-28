# app/main.py
from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.deps import load_meta, load_model
from app.schemas import (
    PredictBatchRequest,
    PredictRequest,
    PredictResponse,
)
from db.crud import log_prediction_io
from db.database import get_db

# ---------- Config ----------
logger = logging.getLogger(__name__)
APP_VERSION = "0.1.0"

MODEL_META = load_meta()
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", MODEL_META.get("threshold", 0.5)))
FEATURE_NAMES = MODEL_META.get("feature_names", None)


def _align(X: pd.DataFrame) -> pd.DataFrame:
    if isinstance(FEATURE_NAMES, list):
        return X.reindex(columns=FEATURE_NAMES)
    return X


app = FastAPI(
    title="Futurisys Turnover API",
    description="POC Projet 5 : API FastAPI (turnover)",
    version=APP_VERSION,
)


def get_model():
    try:
        return load_model()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {e}",
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Model load error: {e}",
        )


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/meta")
def get_meta():
    return {
        "version": APP_VERSION,
        "threshold": DEFAULT_THRESHOLD,
        "model_meta": MODEL_META,
        "env": {
            "HF_MODEL_REPO": os.getenv("HF_MODEL_REPO"),
            "HF_REVISION": os.getenv("HF_REVISION"),
        },
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "version": APP_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict_one(
    req: PredictRequest,
    model: Any = Depends(get_model),
    db: Session = Depends(get_db),
):
    X = _align(pd.DataFrame([req.features]))
    try:
        proba = float(model.predict_proba(X)[0, 1])
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Model has no predict_proba. Check your pipeline.",
        )
    label = int(proba >= DEFAULT_THRESHOLD)

    try:
        log_prediction_io(
            db,
            model_version=APP_VERSION,
            threshold=DEFAULT_THRESHOLD,
            payload=dict(req.features),
            proba=proba,
            label=label,
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("DB log failed (predict_one)", exc_info=err)

    return PredictResponse(
        proba=proba,
        label=label,
        threshold=DEFAULT_THRESHOLD,
    )


@app.post("/predict_batch")
def predict_batch(
    req: PredictBatchRequest,
    model: Any = Depends(get_model),
    db: Session = Depends(get_db),
):
    if len(req.rows) == 0:
        return {"items": []}

    X = _align(pd.DataFrame(req.rows))
    try:
        probas = model.predict_proba(X)[:, 1]
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Model has no predict_proba. Check your pipeline.",
        )

    labels = (probas >= DEFAULT_THRESHOLD).astype(int).tolist()

    try:
        for row, p, lbl in zip(req.rows, probas, labels):
            log_prediction_io(
                db,
                model_version=APP_VERSION,
                threshold=DEFAULT_THRESHOLD,
                payload=dict(row),
                proba=float(p),
                label=int(lbl),
            )
    except Exception as err:  # noqa: BLE001
        logger.warning("DB log failed (predict_batch)", exc_info=err)

    items = [{"proba": float(p), "label": int(lbl)} for p, lbl in zip(probas, labels)]

    return {
        "threshold": DEFAULT_THRESHOLD,
        "items": items,
    }


@app.post("/predict_csv")
async def predict_csv(
    file: UploadFile = File(...),
    model: Any = Depends(get_model),
    db: Session = Depends(get_db),
):
    import io

    raw = await file.read()
    try:
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(raw.decode("cp1252")))

    target_col = "a_quitte_l_entreprise"
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV vide.")

    X = _align(df)
    try:
        probas = model.predict_proba(X)[:, 1]
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Model has no predict_proba. Check your pipeline.",
        )

    labels = (probas >= DEFAULT_THRESHOLD).astype(int).tolist()

    try:
        rows = df.to_dict(orient="records")
        for row, p, lbl in zip(rows, probas, labels):
            log_prediction_io(
                db,
                model_version=APP_VERSION,
                threshold=DEFAULT_THRESHOLD,
                payload=dict(row),
                proba=float(p),
                label=int(lbl),
            )
    except Exception as err:  # noqa: BLE001
        logger.warning("DB log failed (predict_csv)", exc_info=err)

    items = [{"proba": float(p), "label": int(lbl)} for p, lbl in zip(probas, labels)]

    return {
        "filename": file.filename,
        "threshold": DEFAULT_THRESHOLD,
        "n_rows": len(items),
        "items": items,
    }
