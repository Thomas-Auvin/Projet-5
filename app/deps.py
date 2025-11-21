# app/deps.py
import os
import json
from functools import lru_cache
import joblib

MODEL_PATH = os.getenv("MODEL_PATH", "ml/model.joblib")
META_PATH = os.getenv("MODEL_META_PATH", "ml/model_meta.json")


@lru_cache
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache
def load_meta() -> dict:
    """
    Charge le JSON de méta en UTF-8, 
    et si ça échoue (fichiers anciens Windows),
    essaie un fallback CP-1252. Retourne {} si rien ne marche.
    """
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except UnicodeDecodeError:
        try:
            with open(META_PATH, "r", encoding="cp1252") as f:
                return json.load(f)
        except Exception:
            return {}
    except Exception:
        return {}
