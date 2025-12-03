# scripts/seed_db.py
from __future__ import annotations

import os
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from db.crud import log_prediction_io
from db.models import PredInput

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local.db")


def main() -> None:
    """
    Idempotent-ish: si on trouve déjà une ligne
    PredInput qui contient un payload
    avec la clé "seed", on évite de réinsérer.
    """
    engine = create_engine(DATABASE_URL, future=True)
    try:
        with Session(engine) as s:
            # Garde-fou simple : si on a déjà ≥ 2 PredInput,
            # on considère que le seed a déjà été fait
            n_in = s.scalar(select(func.count()).select_from(PredInput)) or 0
            if n_in >= 2:
                print("Seed déjà présent (>=2 inputs) — skip.")
                return

            # 2 lignes de seed via l’API centralisée
            log_prediction_io(
                s,
                model_version="local-dev",
                threshold=0.5,
                payload={"seed": True, "index": 1},
                proba=0.2,
                label=0,
            )
            log_prediction_io(
                s,
                model_version="local-dev",
                threshold=0.5,
                payload={"seed": True, "index": 2},
                proba=0.8,
                label=1,
            )
            s.commit()
            print("Seed OK")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
