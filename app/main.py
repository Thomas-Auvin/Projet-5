# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from typing import Any
import os
import pandas as pd

from app.schemas import (
    PredictRequest,
    PredictBatchRequest,
    PredictResponse,
)
from app.deps import load_model, load_meta

from sqlalchemy.orm import Session
from db.database import get_db
from db.crud import log_prediction_io
from fastapi import UploadFile, File

from fastapi.responses import RedirectResponse

 
# ---------- Config ----------
APP_VERSION = "0.1.0"
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

meta = load_meta()
DEFAULT_THRESHOLD = float(
    os.getenv("THRESHOLD", meta.get("threshold", 0.5))
)
FEATURE_NAMES = meta.get("feature_names", None)


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
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model load error: {e}",
        )


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    return {"status": "ok", "version": APP_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict_one(
    req: PredictRequest,
    model: Any = Depends(get_model),
    # injecte la session DB
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

    # Log en base (ne casse pas la réponse si le log échoue)
    try:
        log_prediction_io(
            db,
            model_version=APP_VERSION,
            threshold=DEFAULT_THRESHOLD,
            payload=dict(req.features),
            proba=proba,
            label=label,
        )
    except Exception:
        pass

    return PredictResponse(
        proba=proba,
        label=label,
        threshold=DEFAULT_THRESHOLD,
    )


@app.post("/predict_batch")
def predict_batch(
    req: PredictBatchRequest,
    model: Any = Depends(get_model),
    # injecte la session DB
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

    # Log chaque ligne
    try:
        for row, p, l in zip(req.rows, probas, labels):
            log_prediction_io(
                db,
                model_version=APP_VERSION,
                threshold=DEFAULT_THRESHOLD,
                payload=dict(row),
                proba=float(p),
                label=int(l),
            )
    except Exception:
        pass

    items = [
        {"proba": float(p), "label": int(l)}
        for p, l in zip(probas, labels)
    ]

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

    # Lecture robuste du CSV (UTF-8 puis fallback CP-1252)
    raw = await file.read()
    try:
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(raw.decode("cp1252")))

    # On ignore la cible si elle est présente
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

    # Log en base (best-effort)
    try:
        for row, p, l in zip(
            df.to_dict(orient="records"), probas, labels
        ):
            log_prediction_io(
                db,
                model_version=APP_VERSION,
                threshold=DEFAULT_THRESHOLD,
                payload=dict(row),
                proba=float(p),
                label=int(l),
            )
    except Exception:
        pass

    items = [
        {"proba": float(p), "label": int(l)}
        for p, l in zip(probas, labels)
    ]

    return {
        "filename": file.filename,
        "threshold": DEFAULT_THRESHOLD,
        "n_rows": len(items),
        "items": items,
    }
