import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    classification_report, confusion_matrix,
    confusion_matrix, roc_auc_score, 
    average_precision_score, precision_recall_curve,
    precision_recall_fscore_support,
)


# ===============================================================
# 1) Utilitaires (seuils & évaluations)
# ===============================================================
def get_positive_scores(model, X):
    """Retourne un score pour la classe positive (1): predict_proba[:,1] sinon decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X)
        if s.ndim == 1:
            return s
        # Multi-sorties: essaye d'attraper la colonne de la classe 1
        pos_idx = -1
        if hasattr(model, "classes_"):
            cls = np.array(model.classes_)
            idx = np.where(cls == 1)[0]
            if len(idx):
                pos_idx = int(idx[0])
        return s[:, pos_idx]
    else:
        raise AttributeError("Le modèle ne fournit ni predict_proba ni decision_function.")

def evaluate_at_threshold(y_true, scores, thr):
    y_pred = (scores >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    ap = average_precision_score(y_true, scores)
    roc = roc_auc_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred)  # [[tn, fp],[fn, tp]]
    rep = classification_report(y_true, y_pred, digits=3, zero_division=0)
    return {
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "avg_precision_pr": float(ap),
        "roc_auc": float(roc),
        "confusion_matrix": cm,
        "report": rep
    }

def pick_threshold(y_true, scores, mode="max_f1", recall_target=0.75):
    """
    Choisit un seuil depuis la courbe PR (F1-max ou rappel cible), via precision_recall_curve (sklearn).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    # Alignement: thresholds a une longueur len(precisions)-1
    P, R, T = precisions[:-1], recalls[:-1], thresholds
    F1 = np.where((P + R) > 0, 2 * P * R / (P + R), 0.0)

    if mode == "max_f1":
        idx = int(np.nanargmax(F1))
    elif mode == "target_recall":
        feas = np.where(R >= recall_target)[0]
        idx = int(feas[np.nanargmax(F1[feas])]) if len(feas) else int(np.nanargmax(F1))
    else:
        raise ValueError("mode doit être 'max_f1' ou 'target_recall'")

    return float(T[idx]), {
        "chosen_mode": mode,
        "recall_target": recall_target if mode == "target_recall" else None,
        "threshold": float(T[idx]),
        "precision": float(P[idx]),
        "recall": float(R[idx]),
        "f1": float(F1[idx]),
        "avg_precision_pr": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }

def pick_threshold_oof(model, X_train, y_train, mode="max_f1", recall_target=0.75, n_splits=5, random_state=42):
    """
    Choix de seuil sur des scores OOF (out-of-fold) pour éviter toute fuite sur le test.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_scores = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thr, summary = pick_threshold(y_train, oof_scores, mode=mode, recall_target=recall_target)
    return float(thr), summary
