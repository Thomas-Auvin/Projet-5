#!/usr/bin/env bash
set -euo pipefail

echo "[start] PORT=${PORT:-7860}"
echo "[start] DATABASE_URL=${DATABASE_URL:-sqlite:////data/projet5.db}"

mkdir -p ml

# --- Téléchargement du modèle depuis Hugging Face si HF_MODEL_REPO est défini ---
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = os.getenv("HF_MODEL_REPO")  # ex: "Thomas-Auvin/projet5-turnover-model"
model_fname = os.getenv("HF_MODEL_FILENAME", "model.joblib")
meta_fname  = os.getenv("HF_MODEL_META_FILENAME", "model_meta.json")

target_model = Path(os.getenv("MODEL_PATH", "ml/model.joblib"))
target_meta  = Path(os.getenv("MODEL_META_PATH", "ml/model_meta.json"))

force = os.getenv("FORCE_DOWNLOAD", "0") == "1"

def need(p: Path) -> bool:
    return force or (not p.exists() or p.stat().st_size == 0)

if repo:
    if need(target_model):
        p = hf_hub_download(repo_id=repo, filename=model_fname,
                            local_dir="ml", local_dir_use_symlinks=False)
        Path(p).rename(target_model) if Path(p) != target_model else None
        print(f"[start] downloaded {model_fname} -> {target_model}")
    if need(target_meta):
        p = hf_hub_download(repo_id=repo, filename=meta_fname,
                            local_dir="ml", local_dir_use_symlinks=False)
        Path(p).rename(target_meta) if Path(p) != target_meta else None
        print(f"[start] downloaded {meta_fname} -> {target_meta}")
else:
    print("[start] HF_MODEL_REPO not set; expecting local ml/ files.")
PY

# DB par défaut (Space) : SQLite persistant dans /data
export DATABASE_URL="${DATABASE_URL:-sqlite:////data/projet5.db}"

# Lancer l'API
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-7860}"
