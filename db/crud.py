# db/crud.py
from __future__ import annotations
from sqlalchemy.orm import Session
from .models import PredInput, PredOutput


def log_prediction_io(
    db: Session,
    *,
    model_version: str,
    threshold: float,
    payload: dict,
    proba: float,
    label: int,
) -> str:
    """
    Insère une trace d'input puis d'output.
    Retourne l'UID (PredInput.uid) permettant de recoller les deux.
    """
    pi = PredInput(
        model_version=model_version,
        threshold=threshold,
        payload=payload
        )
    db.add(pi)
    db.flush()  # génère pi.uid

    po = PredOutput(
        input_uid=pi.uid,
        proba=proba,
        label=label,
        served_by="api"
        )
    db.add(po)
    db.commit()
    return pi.uid
