# ml/train_model.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import sys

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from ml.pipeline import build_model

# -------------------- Paramètres généraux --------------------
RANDOM_STATE = 42
USE_CALIBRATION = True
CALIB_METHOD = "isotonic"  # "isotonic" ou "sigmoid"
THRESHOLD_FINAL = 0.1480  # ← ton seuil choisi en P4 (modifie si besoin)
TARGET_COL = "a_quitte_l_entreprise"

DATA_PATH = Path("data/train.csv")  # adapte au chemin réel
MODEL_PATH = Path("ml/model.joblib")
META_PATH = Path("ml/model_meta.json")

# Si tu as une liste de colonnes à supprimer :
COLS_TO_DROP: list[str] = []


# -------------------- Utilitaires --------------------
def get_positive_scores(model, X):
    """Retourne le score proba de la classe positive (1)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return s if s.ndim == 1 else s[:, -1]
    raise AttributeError("Le modèle ne fournit ni predict_proba ni decision_function.")


def evaluate_at_threshold(y_true, scores, thr):
    """Rapport complet @seuil (pour information)."""
    y_pred = (scores >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    ap = average_precision_score(y_true, scores)
    roc = roc_auc_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred).tolist()
    rep = classification_report(y_true, y_pred, digits=3, zero_division=0)
    return dict(
        threshold=float(thr),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        accuracy=float(acc),
        balanced_accuracy=float(bacc),
        avg_precision_pr=float(ap),
        roc_auc=float(roc),
        confusion_matrix=cm,
        report=rep,
    )


# -------------------- Entraînement principal --------------------
def main():
    # 1) Charger les données
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"DATA_PATH introuvable: {DATA_PATH}. Placer le CSV ici.")
    df = pd.read_csv(DATA_PATH)

    # 2) Cible binaire 0/1
    if df[TARGET_COL].dtype == "O":
        y = df[TARGET_COL].map({"Non": 0, "Oui": 1}).astype(int)
    else:
        y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # 3) Split (test uniquement pour reporting local)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # 4) Pipeline figé (hyperparams issus de ton GridSearch P4)
    pipe = build_model(cols_to_drop=COLS_TO_DROP, random_state=RANDOM_STATE)

    # (optionnel) si tu veux fixer scale_pos_weight selon y_train :
    # spw = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    # pipe.set_params(xgb__scale_pos_weight=spw)

    # 5) Fit sur TRAIN
    pipe.fit(X_train, y_train)

    # 6) Calibration (comme en P4)
    if USE_CALIBRATION:
        calibrator = CalibratedClassifierCV(estimator=pipe, method=CALIB_METHOD, cv=5)
        calibrator.fit(X_train, y_train)
        final_model = calibrator
    else:
        final_model = pipe

    # 7) Reporting local (seuil-indépendant + @seuil figé)
    proba_test = get_positive_scores(final_model, X_test)
    print("\n=== AUC/AP (test) ===")
    print("ROC AUC :", f"{roc_auc_score(y_test, proba_test):.3f}")
    print("Avg Precision (PR AUC) :", f"{average_precision_score(y_test, proba_test):.3f}")

    res = evaluate_at_threshold(y_test, proba_test, THRESHOLD_FINAL)
    print(f"\n=== Rapport @ seuil {THRESHOLD_FINAL:.4f} ===")
    print(res["report"])
    print("Confusion matrix [[tn, fp],[fn, tp]]:", res["confusion_matrix"])

    # 8) Sauvegardes (modèle + méta)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODEL_PATH)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
        "threshold": THRESHOLD_FINAL,
        "target": TARGET_COL,
        "feature_names": X.columns.tolist(),
        "calibrated": USE_CALIBRATION,
        "calibration_method": CALIB_METHOD if USE_CALIBRATION else None,
        "random_state": RANDOM_STATE,
        "note": "Hyperparamètres XGB figés (issus du GridSearch P4).",
    }

    # ⚠️ ECRITURE UTF-8 POUR EVITER L'ERREUR UnicodeDecodeError
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved model to:   {MODEL_PATH.resolve()}")
    print(f"Saved metadata to:{META_PATH.resolve()}")


if __name__ == "__main__":
    main()
