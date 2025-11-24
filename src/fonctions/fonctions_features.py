import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Reproduit ton feature engineering sans fuite de données.
    - Les statistiques par groupe (médianes salaire) sont apprises en `fit` sur le TRAIN.
    - Elles sont réutilisées en `transform` sur les folds / le test.
    """

    def __init__(self, cast_flags: bool = True):
        # option pour caster les flags en numériques (0/1) en sortie
        self.cast_flags = cast_flags

        # stats apprises sur le TRAIN
        self.dept_med_ = None
        self.niv_med_ = None
        self.global_med_ = None

        # colonnes utilisées telles que nommées
        self.sat_cols = [
            "satisfaction_employee_environnement",
            "satisfaction_employee_nature_travail",
            "satisfaction_employee_equipe",
            "satisfaction_employee_equilibre_pro_perso",
        ]
        self.cols_poste = ["annees_dans_l_entreprise", "annees_dans_le_poste_actuel"]

        # flags à caster (0/1) pour éviter OHE involontaire côté préprocessing
        self.flag_cols_ = [
            "perf_degrade_flag",
            "hors_entreprise_majoritaire",
            "a_connu_mvmnt_interne",
            "long_commute_flag",
            "recent_promo_flag",
            "pee_participation_flag",
            "pee_participation_2plus",
            "is_manager_flag",
            "heure_supp_flag",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        """Apprend uniquement les médianes nécessaires sur le TRAIN."""
        df = X.copy()

        # Typages minimaux nécessaires pour apprendre les médianes
        df["revenu_mensuel"] = pd.to_numeric(df.get("revenu_mensuel"), errors="coerce")
        df["niveau_hierarchique_poste"] = pd.to_numeric(
            df.get("niveau_hierarchique_poste"), errors="coerce"
        )

        # Médiane globale (fallback pour catégories jamais vues)
        self.global_med_ = df["revenu_mensuel"].median()

        # Médianes apprises sur TRAIN uniquement
        # (groupby sur colonnes présentes ; si manquantes, ça lèvera plus tôt et de façon claire)
        self.dept_med_ = df.groupby("departement")["revenu_mensuel"].median()
        self.niv_med_ = df.groupby("niveau_hierarchique_poste")["revenu_mensuel"].median()

        return self  # impératif pour sklearn

    def transform(self, X: pd.DataFrame):
        """Génère les features, puis (optionnel) caste les flags en numériques."""
        df = X.copy()

        # Typages nécessaires (robustesse)
        df["revenu_mensuel"] = pd.to_numeric(df.get("revenu_mensuel"), errors="coerce")
        df["niveau_hierarchique_poste"] = pd.to_numeric(
            df.get("niveau_hierarchique_poste"), errors="coerce"
        )

        # -- 1) delta & flags perf
        df["delta_note_evaluation"] = (
            df["note_evaluation_precedente"] - df["note_evaluation_actuelle"]
        )
        df["perf_degrade_flag"] = df["delta_note_evaluation"] > 0
        df["perf_degrade_niv"] = np.where(
            df["delta_note_evaluation"] > 0, df["delta_note_evaluation"], 0
        )

        # -- 2) années d'expérience hors entreprise
        df["nb_annnee_hors_entreprise"] = (
            df["annee_experience_totale"] - df["annees_dans_l_entreprise"]
        ).clip(lower=0)

        df["ratio_dans_et_hors_entreprise"] = np.where(
            df["annee_experience_totale"] > 0,
            df["nb_annnee_hors_entreprise"] / df["annee_experience_totale"],
            np.nan,
        )
        df["hors_entreprise_majoritaire"] = df["ratio_dans_et_hors_entreprise"] >= 0.5

        # -- 3) années hors poste actuel + mouvement interne
        df[self.cols_poste] = df[self.cols_poste].apply(pd.to_numeric, errors="coerce")
        df["Ecart_nb_annee_sur_poste"] = (
            df["annees_dans_l_entreprise"] - df["annees_dans_le_poste_actuel"]
        ).clip(lower=0)

        valid = df[self.cols_poste].notna().all(axis=1)
        df["a_connu_mvmnt_interne"] = df["Ecart_nb_annee_sur_poste"].gt(0).where(valid, pd.NA)

        df["tenure_ratio_current_post"] = np.where(
            df["annees_dans_l_entreprise"] > 0,
            df["annees_dans_le_poste_actuel"] / df["annees_dans_l_entreprise"],
            np.nan,
        )

        # -- 4) agrégats satisfaction
        df["sat_mean"] = df[self.sat_cols].mean(axis=1)
        df["sat_min"] = df[self.sat_cols].min(axis=1)
        df["sat_std"] = df[self.sat_cols].std(axis=1)
        df["sat_low_flag"] = (df[self.sat_cols] <= 2).sum(axis=1) >= 2

        # -- 5) contexte salarial (avec médianes apprises au fit)
        med_dept = df["departement"].map(self.dept_med_).fillna(self.global_med_)
        med_niv = df["niveau_hierarchique_poste"].map(self.niv_med_).fillna(self.global_med_)
        df["revenu_vs_dept_med"] = df["revenu_mensuel"] / med_dept
        df["revenu_vs_niveau_med"] = df["revenu_mensuel"] / med_niv

        # -- 6) heures supp & déplacements
        df["heure_supp_flag"] = df["heure_supplementaires"].map({"Oui": 1, "Non": 0})
        df["long_commute_flag"] = df["distance_domicile_travail"] >= 15

        # -- 7) intensité formation
        df["formations_par_an"] = np.where(
            df["annees_dans_l_entreprise"] > 0,
            df["nb_formations_suivies"] / df["annees_dans_l_entreprise"],
            np.nan,
        )

        # -- 8) promotions récentes
        df["recent_promo_flag"] = df["annees_depuis_la_derniere_promotion"] <= 1

        # -- 9) PEE
        df["pee_participation_flag"] = df["nombre_participation_pee"] > 0
        df["pee_participation_2plus"] = df["nombre_participation_pee"] >= 2

        # -- 10) statut manager
        df["is_manager_flag"] = df["niveau_hierarchique_poste"] >= 3

        # -- 11)suppression des colonnes après la permutation importance 

        #df = df.drop(columns=[
        #    "heure_supplementaires",
        #    "perf_degrade_flag",
        #    "pee_participation_flag",
        #    "pee_participation_2plus",
        #    "is_manager_flag",
        #    "poste",
        #    "domaine_etude",          
        #    "frequence_deplacement"
        #    ],
        #        errors="ignore",
        #    )
        # -- 11) Caster les flags en numérique (0/1) pour les traiter comme numériques
        if self.cast_flags:
            for col in self.flag_cols_:
                if col in df.columns:  # garde-fou
                    df[col] = df[col].astype(float)

        return df


__all__ = ["FeatureEngineer"]

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        # ne PAS transformer le paramètre ici (pas de list(), pas de copy)
        self.columns = columns  # peut être list/tuple/None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = list(self.columns) if self.columns is not None else []
        return X.drop(columns=cols, errors="ignore")

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([])
        cols_set = set(self.columns or [])
        return np.array([c for c in input_features if c not in cols_set])

