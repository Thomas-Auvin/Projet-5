# scripts/db_check.py
from __future__ import annotations

from sqlalchemy import select, func
from db.database import SessionLocal
from db.models import PredInput, PredOutput


def main() -> None:
    # Ouvre une session SQLAlchemy
    with SessionLocal() as db:
        # Compteurs
        n_in = db.scalar(select(func.count()).select_from(PredInput)) or 0
        n_out = db.scalar(select(func.count()).select_from(PredOutput)) or 0
        print(f"PredInput rows: {n_in}  |  PredOutput rows: {n_out}")

        # Derniers enregistrements
        latest_inputs = db.scalars(select(PredInput).order_by(PredInput.id.desc()).limit(5)).all()
        latest_outputs = db.scalars(
            select(PredOutput).order_by(PredOutput.id.desc()).limit(5)
        ).all()

    print("\nDerniers PredInput:")
    for r in latest_inputs:
        payload_preview = str(r.payload)[:120].replace("\n", " ")
        print(f"{r.id}  {r.uid}  {r.created_at}  thr={r.threshold}  {payload_preview}")

    print("\nDerniers PredOutput:")
    for r in latest_outputs:
        print(
            f"{r.id}  input_uid={r.input_uid}  {r.created_at}  proba={r.proba:.3f}  label={r.label}"
        )


if __name__ == "__main__":
    main()
