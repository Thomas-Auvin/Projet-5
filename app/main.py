# app/main.py
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from typing import Any
import os
import pandas as pd

from app.schemas import PredictRequest, PredictBatchRequest, PredictResponse
from app.deps import load_model, load_meta

import io  # pour lire le CSV en mémoire


# ---------- Config ----------
APP_VERSION = "0.1.0"
meta = load_meta()
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", meta.get("threshold", 0.5)))
FEATURE_NAMES = meta.get("feature_names", None)


def _align(X: pd.DataFrame) -> pd.DataFrame:
    if isinstance(FEATURE_NAMES, list):
        return X.reindex(columns=FEATURE_NAMES)
    return X


app = FastAPI(
    title="Futurisys Turnover API",
    description="POC Projet 5 : Modèle du Projet 4 (turnover)",
    version=APP_VERSION,
)


def get_model():
    try:
        return load_model()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {e}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")


@app.get("/health")
def health_check():
    return {"status": "ok", "version": APP_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict_one(req: PredictRequest, model: Any = Depends(get_model)):
    # DataFrame d’une seule ligne
    X = _align(pd.DataFrame([req.features]))
    # On suppose un pipeline sklearn avec .predict_proba
    try:
        proba = float(model.predict_proba(X)[0, 1])
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Model has no predict_proba. Check your pipeline."
            )
    label = int(proba >= DEFAULT_THRESHOLD)
    return PredictResponse(
        proba=proba, label=label,
        threshold=DEFAULT_THRESHOLD
        )


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest, model: Any = Depends(get_model)):
    if len(req.rows) == 0:
        return {"items": []}
    X = _align(pd.DataFrame(req.rows))
    try:
        probas = model.predict_proba(X)[:, 1]
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Model has no predict_proba. Check your pipeline."
            )
    labels = (probas >= DEFAULT_THRESHOLD).astype(int).tolist()
    return {
        "threshold": DEFAULT_THRESHOLD,
        "items": [
            {"proba": float(p),
             "label": int(l)} for p,
            l in zip(probas, labels)
            ]
    }


@app.post("/predict_csv")
async def predict_csv(
    file: UploadFile = File(
        ..., description="Fichier CSV avec une ligne par personne"
        ),
    sep: str = ",",  # au besoin, tu peux passer sep=";" pour des CSV FR
    model: Any = Depends(get_model),
):
    # 1) Lire le fichier (on essaye UTF-8 puis fallback CP-1252)
    raw = await file.read()
    df = None
    for enc in ("utf-8", "cp1252"):
        try:
            df = pd.read_csv(io.StringIO(raw.decode(enc)), sep=sep)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise HTTPException(
            status_code=400,
            detail="Impossible de décoder le CSV (essaie UTF-8 ou CP-1252).",
        )

    # 2) Si la colonne cible existe,
    #    on la retire (le modèle n'en a pas besoin pour prédire)
    target = meta.get("target")
    if target and target in df.columns:
        df = df.drop(columns=[target])

    # 3) Aligner l'ordre / l'ensemble
    #    des colonnes sur celles du modèle (FEATURE_NAMES)
    X = _align(df)

    # 4) Prédictions
    try:
        probas = model.predict_proba(X)[:, 1]
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Model has no predict_proba. Check your pipeline."
            )

    labels = (probas >= DEFAULT_THRESHOLD).astype(int).tolist()
    items = [
        {"proba": float(p), "label": int(l)} for p, l in zip(probas, labels)
        ]
    return {"threshold": DEFAULT_THRESHOLD, "items": items}
