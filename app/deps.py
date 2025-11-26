# app/deps.py
import os
import json
from functools import lru_cache
from pathlib import Path
import joblib

# ↓ nouvelle import
from huggingface_hub import hf_hub_download

# Emplacements locaux (si déjà présents on les utilise)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "ml/model.joblib"))
META_PATH = Path(os.getenv("MODEL_META_PATH", "ml/model_meta.json"))

# Réglages du repo modèle sur HF (où tu viens d’uploader les artefacts)
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Thomas-Auvin/projet5-turnover-model")
HF_MODEL_FILE = os.getenv(
    "HF_MODEL_FILE", "model.joblib"
)  # ou "ml/model.joblib" si tu as gardé le sous-dossier
HF_META_FILE = os.getenv("HF_META_FILE", "model_meta.json")  # idem
HF_REVISION = os.getenv("HF_REVISION")  # optionnel: tag/commit/branch
HF_TOKEN = os.getenv("HF_TOKEN")  # obligatoire si le repo modèle est privé

# Dossier cache (persistant sur HF Spaces : /data)
CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/data/models")).resolve()


def _ensure_download(local_path: Path, repo_id: str, filename: str) -> Path:
    """
    Si local_path n'existe pas, télécharge depuis HF Hub dans CACHE_DIR
    et retourne le chemin local téléchargé.
    """
    if local_path.exists():
        return local_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=HF_REVISION,
        token=HF_TOKEN,
        local_dir=str(CACHE_DIR),
        local_dir_use_symlinks=False,  # fichier réel, pas symlink
    )
    return Path(downloaded)


@lru_cache
def load_model():
    # essaie d’abord MODEL_PATH ; sinon, charge depuis HF
    local = (
        MODEL_PATH
        if MODEL_PATH.exists()
        else _ensure_download(MODEL_PATH, HF_MODEL_REPO, HF_MODEL_FILE)
    )
    return joblib.load(local)


@lru_cache
def load_meta() -> dict:
    try:
        local = (
            META_PATH
            if META_PATH.exists()
            else _ensure_download(META_PATH, HF_MODEL_REPO, HF_META_FILE)
        )
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except UnicodeDecodeError:
        # fallback CP-1252 si fichier ancien Windows
        try:
            with open(local, "r", encoding="cp1252") as f:
                return json.load(f)
        except Exception:
            return {}
    except Exception:
        return {}
