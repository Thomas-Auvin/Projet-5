from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

def get_transformed_feature_names_and_source_map(pipe, X_sample=None, y_sample=None):
    """
    Retourne:
      - feat_names: noms finaux des features (get_feature_names_out du ColumnTransformer)
      - source_map: dict "feature_transformée" -> "colonne source"
    Si 'prep' n'est pas fit, on fit un clone de fe+prep sur (X_sample, y_sample).
    """
    prep = pipe.named_steps["prep"]

    # S'assure que 'prep' est fit
    try:
        feat_names = prep.get_feature_names_out()
        prep_fitted = prep
    except Exception:
        if X_sample is None:
            raise ValueError(
                "Le preprocess n'est pas fit. "
                "Passe X_sample (et éventuellement y_sample) ou fit ton pipeline avant l'appel."
            )
        fe_prep = Pipeline(pipe.steps[:-1])  # (fe + prep)
        fe_prep_fitted = clone(fe_prep).fit(X_sample, y_sample)
        prep_fitted = fe_prep_fitted.named_steps["prep"]
        feat_names = prep_fitted.get_feature_names_out()

    # --- NUMÉRIQUES ---
    num_map = {}
    if "num" in prep_fitted.named_transformers_:
        num_pipe = prep_fitted.named_transformers_["num"]
        # Noms d'entrée réellement utilisés par l'imputer
        num_in = num_pipe.named_steps["imputer"].feature_names_in_.tolist()
        num_map = {f"num__{c}": c for c in num_in}

    # --- CATÉGORIELS ---
    cat_map = {}
    if "cat" in prep_fitted.named_transformers_:
        cat_pipe = prep_fitted.named_transformers_["cat"]
        ohe = cat_pipe.named_steps["ohe"]
        # Colonnes d'entrée de l'OHE : selon ta version de sklearn
        try:
            cat_in = ohe.feature_names_in_.tolist()
        except AttributeError:
            cat_in = cat_pipe.named_steps["imputer"].feature_names_in_.tolist()

        # ohe.categories_ est aligné avec cat_in
        for col, cats in zip(cat_in, ohe.categories_):
            for cat in cats:
                cat_map[f"cat__{col}_{cat}"] = col

    # Fusion + fallback
    source_map = {**num_map, **cat_map}

    def fallback(name):
        if name in source_map:
            return source_map[name]
        if name.startswith("num__"):
            return name.replace("num__", "", 1)
        if name.startswith("cat__"):
            return name.replace("cat__", "", 1).split("_", 1)[0]
        return name

    source_map = {n: source_map.get(n, fallback(n)) for n in feat_names}
    return feat_names, source_map


def perm_importance_cv(pipe, X, y, scoring="average_precision", n_splits=5, n_repeats=10, random_state=42):
    """
    Calcule la permutation importance en CV sur les features transformées.
    Retourne un DataFrame avec mean/std par feature.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fe_prep = Pipeline(pipe.steps[:-1])            # fe + preprocess
    clf_base = pipe.named_steps[list(pipe.named_steps.keys())[-1]]  # le dernier (LogReg ici)

    all_importances = []     # liste d'arrays [n_features] pour chaque fold
    feat_names_ref = None

    for tr, va in cv.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        # fit fe+prep sur TRAIN, transforme
        fe_prep_fit = clone(fe_prep).fit(Xtr, ytr)
        Xtr_t = fe_prep_fit.transform(Xtr)
        Xva_t = fe_prep_fit.transform(Xva)

        # noms de features transformées (depuis le ColumnTransformer)
        feat_names = fe_prep_fit.named_steps["prep"].get_feature_names_out()
        if feat_names_ref is None:
            feat_names_ref = feat_names
        else:
            # sécurité : on s'assure que l’ordre est le même à chaque fold
            assert list(feat_names_ref) == list(feat_names), "Incohérence de noms de features entre folds."

        # fit clf sur transformées
        clf = clone(clf_base).fit(Xtr_t, ytr)

        # permutation importance sur Xva_t
        perm = permutation_importance(
            clf, Xva_t, yva,
            scoring=scoring, n_repeats=n_repeats,
            random_state=random_state, n_jobs=-1
        )
        all_importances.append(perm.importances_mean)

    importances = np.vstack(all_importances)  # shape: (n_folds, n_features)
    df_perm = (pd.DataFrame({
        "feature": feat_names_ref,
        "imp_mean": importances.mean(axis=0),
        "imp_std": importances.std(axis=0)
    }).sort_values("imp_mean", ascending=False))

    return df_perm
