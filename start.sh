#!/usr/bin/env bash
set -euo pipefail

echo "===== Application Startup at $(date) ====="
echo "[start] PORT=${PORT:-7860}"
echo "[start] DATABASE_URL=${DATABASE_URL:-<not-set>}"

# Rendre le token HF visible par huggingface_hub
export HUGGINGFACEHUB_API_TOKEN="${HF_TOKEN:-${HUGGINGFACEHUB_API_TOKEN:-}}"

# Chemins locaux (avec fallback)
export MODEL_PATH="${MODEL_PATH:-ml/model.joblib}"
export MODEL_META_PATH="${MODEL_META_PATH:-ml/model_meta.json}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/data/model_cache}"

# Télécharger le modèle depuis HF si un repo est fourni
python - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = os.getenv("HF_MODEL_REPO")         # ex: "Thomas-Auvin/projet5-turnover-model"
model_file = os.getenv("HF_MODEL_FILE", "ml/model.joblib")
meta_file  = os.getenv("HF_META_FILE",  "ml/model_meta.json")

dst_model = Path(os.getenv("MODEL_PATH", "ml/model.joblib"))
dst_meta  = Path(os.getenv("MODEL_META_PATH", "ml/model_meta.json"))
cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "/data/model_cache"))

Path("ml").mkdir(exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

if repo:
    if not dst_model.exists():
        p = hf_hub_download(repo_id=repo, filename=model_file, local_dir=cache_dir)
        shutil.copy2(p, dst_model)
        print(f"[start] downloaded {model_file} -> {dst_model}")
    else:
        print("[start] model already present, skip download")

    if not dst_meta.exists():
        p = hf_hub_download(repo_id=repo, filename=meta_file, local_dir=cache_dir)
        shutil.copy2(p, dst_meta)
        print(f"[start] downloaded {meta_file} -> {dst_meta}")
    else:
        print("[start] meta already present, skip download")
else:
    print("[start] HF_MODEL_REPO not set — assuming files are baked into the image.")
PY

# Créer / mettre à jour les tables (SQLite ou Postgres selon DATABASE_URL)
python -m db.create_db || true

# Lancer l'API
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-7860}"
