# app/deps.py
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib
from huggingface_hub import hf_hub_download

# -Chemins locaux par défaut (si les fichiers existent déjà, on les utilise)-
MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", "ml/model.joblib"))
META_PATH: Path = Path(os.getenv("MODEL_META_PATH", "ml/model_meta.json"))

# --- Réglages du repo modèle sur Hugging Face Hub ---
HF_MODEL_REPO: Optional[str] = os.getenv("HF_MODEL_REPO", "Thomas-Auvin/projet5-turnover-model")
HF_MODEL_FILE: str = os.getenv("HF_MODEL_FILE", "model.joblib")
HF_META_FILE: str = os.getenv("HF_META_FILE", "model_meta.json")
HF_REVISION: Optional[str] = os.getenv("HF_REVISION")
# tag/branch/commit (obligatoire côté CI/CD)
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
# requis uniquement si repo privé

# Cache local optionnel (sur HF Spaces, /data persiste entre relances)
MODEL_CACHE_DIR: Optional[str] = os.getenv("MODEL_CACHE_DIR")
# ex: "/data/models"


def _download_from_hub(filename: str, local_dir: Path) -> Path:
    """
    Télécharge `filename` depuis le repo HF dans `local_dir`
    et retourne le chemin local.
    Exige HF_REVISION pour pinner une version (sécurité / reproductibilité).
    """
    if not HF_MODEL_REPO:
        raise RuntimeError("HF_MODEL_REPO manquant (repo du modèle sur Hugging Face Hub).")

    # IMPORTANT : on exige une révision “pinnée”
    # (tag ou SHA) pour éviter les téléchargements flottants.
    if not HF_REVISION:
        raise RuntimeError(
            "HF_REVISION manquant. "
            "Renseigne un tag ou un commit SHA du repo modèle "
            "(ex: 'v1.0.0' ou '3a5f9e8...')."
        )

    local_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        # nosec B615: HF_REVISION est vérifié ci-dessus
        repo_id=HF_MODEL_REPO,
        filename=filename,
        revision=HF_REVISION,  # kw-only
        token=HF_TOKEN,  # kw-only (None si public)
        cache_dir=Path(MODEL_CACHE_DIR) if MODEL_CACHE_DIR else None,
        # kw-only
        local_dir=str(local_dir),  # kw-only: écrit une copie locale
    )
    return Path(downloaded_path)


def _ensure_local_file(target_path: Path, hub_filename: str) -> Path:
    """
    Retourne un chemin local existant vers le fichier :
      - si target_path existe déjà → on l’utilise
      - sinon → téléchargement depuis le Hub dans target_path.parent
    """
    if target_path.exists():
        return target_path
    return _download_from_hub(filename=hub_filename, local_dir=target_path.parent)


@lru_cache
def load_model() -> Any:
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
