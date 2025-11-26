# app/deps.py
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
from huggingface_hub import hf_hub_download

# -Chemins locaux par défaut (si le fichier est déjà présent, on l'utilise)-
MODEL_PATH = Path(os.getenv("MODEL_PATH", "ml/model.joblib"))
META_PATH = Path(os.getenv("MODEL_META_PATH", "ml/model_meta.json"))

# --- Réglages du repo modèle sur Hugging Face Hub ---
HF_MODEL_REPO: Optional[str] = os.getenv(
    "HF_MODEL_REPO",
    "Thomas-Auvin/projet5-turnover-model",
)

# Nom du fichier modèle dans le repo HF
HF_MODEL_FILE: str = os.getenv("HF_MODEL_FILE", "model.joblib")

# Nom du fichier méta dans le repo HF
HF_META_FILE: str = os.getenv("HF_META_FILE", "model_meta.json")

# Tag/branche/commit (optionnel)
HF_REVISION: Optional[str] = os.getenv("HF_REVISION")

# Requis si le repo modèle est privé
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

# Cache local (optionnel). Sur HF Spaces, /data persiste entre relances.
# Ex : "/data/models"
MODEL_CACHE_DIR: Optional[str] = os.getenv("MODEL_CACHE_DIR")


def _download_from_hub(filename: str, local_dir: Path) -> Path:
    """
    Télécharge `filename` depuis le repo HF dans `local_dir`
    et retourne le chemin local.
    Utilise un cache (MODEL_CACHE_DIR) si défini.
    """
    if not HF_MODEL_REPO:
        raise RuntimeError("HF_MODEL_REPO manquant (repo du modèle sur Hugging Face Hub).")

    local_dir.mkdir(parents=True, exist_ok=True)

    # Appel avec kwargs conformes aux stubs (mypy) et API actuelle.
    downloaded_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=filename,
        revision=HF_REVISION,
        token=HF_TOKEN,
        cache_dir=Path(MODEL_CACHE_DIR) if MODEL_CACHE_DIR else None,
        local_dir=str(local_dir),
    )
    return Path(downloaded_path)


def _ensure_local_file(target_path: Path, hub_filename: str) -> Path:
    """
    Retourne un chemin local existant vers le fichier :
      - si target_path existe déjà → on l’utilise
      - sinon → on télécharge depuis le Hub dans target_path.parent
    """
    if target_path.exists():
        return target_path
    return _download_from_hub(filename=hub_filename, local_dir=target_path.parent)


@lru_cache
def load_model():
    """
    Charge l’objet modèle (joblib) :
      - priorité au chemin local MODEL_PATH s’il existe
      - sinon, télécharge depuis le Hub puis charge
    """
    local = _ensure_local_file(MODEL_PATH, HF_MODEL_FILE)
    return joblib.load(local)


@lru_cache
def load_meta() -> dict:
    """
    Charge le JSON de métadonnées :
      - priorité au chemin local META_PATH s’il existe
      - sinon, télécharge depuis le Hub puis charge
      - fallback d’encodage sur CP-1252 si nécessaire
    """
    try:
        local = _ensure_local_file(META_PATH, HF_META_FILE)
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except UnicodeDecodeError:
        try:
            with open(local, "r", encoding="cp1252") as f:
                return json.load(f)
        except Exception:
            return {}
    except Exception:
        return {}
