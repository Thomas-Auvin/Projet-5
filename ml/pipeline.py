# ml/pipeline.py
from __future__ import annotations

from typing import List, Optional
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# ✅ Tes classes (depuis ton projet)
from src.fonctions.fonctions_features import FeatureEngineer, ColumnDropper


# --- IMPORTANT : fonction top-level (remplace le lambda) ----
def to_int8(X):
    """
    Convertit proprement un DataFrame/array de bool -> int8 (0/1).
    Définie au niveau module pour être picklable par joblib.
    """
    try:
        return X.astype(np.int8)
    except Exception:
        return X.astype("int8")


def build_preprocess() -> ColumnTransformer:
    """
    Recrée exactement ton préprocessing P4 :
      - num : imputer(median) + standard scaler
      - bool : cast -> int8 + imputer(most_frequent)
      - cat : imputer(most_frequent) + OneHotEncoder(ignore)
    """
    numeric_cont_sel = selector(dtype_include=["number"], dtype_exclude=["bool"])
    bool_sel = selector(dtype_include=["bool"])
    categorical_sel = selector(dtype_exclude=["number", "bool"])

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # ⚠️ plus de lambda ici : FunctionTransformer(to_int8, ...)
    bool_pipe = Pipeline(
        [
            (
                "to_int",
                FunctionTransformer(to_int8, feature_names_out="one-to-one", validate=False),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cont_sel),
            ("bool", bool_pipe, bool_sel),
            ("cat", cat_pipe, categorical_sel),
        ],
        remainder="drop",
    )
    return preprocess


def build_model(cols_to_drop: Optional[List[str]] = None, random_state: int = 42) -> Pipeline:
    """
    Pipeline final (sans GridSearch) avec hyperparamètres figés
    d'après tes meilleurs résultats P4.
    """
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        # 🔒 hyperparams figés (issus de ton GridSearch P4)
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.7,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=5.0,
        # scale_pos_weight: optionnel, à fixer au training si tu veux
    )

    pipe = Pipeline(
        steps=[
            ("fe", FeatureEngineer()),
            ("drop", ColumnDropper(columns=list(cols_to_drop or []))),
            ("prep", build_preprocess()),
            ("xgb", xgb),
        ]
    )
    return pipe
