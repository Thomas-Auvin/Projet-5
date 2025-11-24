# scripts/ingest_dataset.py
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterator

import pandas as pd

from db.database import SessionLocal
from db.models import DatasetFile, DatasetRow


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iter_chunks(path: Path, chunksize: int = 2000) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path", type=str, help="Chemin vers le CSV à ingérer"
        )
    parser.add_argument("--source", type=str, default="manual")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--chunksize", type=int, default=2000)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    checksum = sha256_file(csv_path)
    print(f"[INFO] checksum: {checksum}")

    # Lire un petit échantillon pour récupérer colonnes + dtypes textuels
    head_df = pd.read_csv(csv_path, nrows=1000)
    cols = list(head_df.columns)
    dtypes = {c: str(head_df[c].dtype) for c in cols}
    meta_cols = {"cols": cols, "dtypes": dtypes}

    total_rows = 0
    with SessionLocal() as db:
        # Dédupe par checksum
        existing = (
            db.query(DatasetFile)
            .filter(DatasetFile.checksum == checksum)
            .one_or_none()
        )
        if existing:
            print("[WARN] Fichier déjà ingéré (checksum identique). Abandon.")
            return

        df_meta = DatasetFile(
            filename=csv_path.name,
            checksum=checksum,
            n_rows=0,
            columns=meta_cols,
            source=args.source,
            notes=args.notes,
        )
        db.add(df_meta)
        db.flush()          # -> df_meta.id est maintenant peuplé par le SGBD
        df_id = df_meta.id  # on fige l'ID tant que la session est ouverte
        db.commit()

        # Insertion par chunks
        row_index_base = 0
        for chunk in iter_chunks(csv_path, chunksize=args.chunksize):
            records = []
            for i, row in enumerate(chunk.to_dict(orient="records")):
                records.append(
                    DatasetRow(
                        file_id=df_id,      # <-- on utilise df_id (solide)
                        row_index=row_index_base + i,
                        payload=row,
                    )
                )
            db.bulk_save_objects(records)
            db.commit()
            total_rows += len(records)
            row_index_base += len(records)
            print(f"[INFO] +{len(records)} lignes (total {total_rows})")

        # Mise à jour du compteur
        df_meta.n_rows = total_rows
        db.add(df_meta)
        db.commit()

    # Hors session : on affiche la valeur figée
    print(f"[DONE] Ingestion terminée: file_id={df_id} rows={total_rows}")


if __name__ == "__main__":
    main()
