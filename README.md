---
title: Projet5 Turnover Api
emoji: 💻
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
short_description: Projet 5 - API sur le turnover
---

# Projet 5 – API de prédiction de turnover (FastAPI + ML + BDD)

Ce dépôt contient le livrable du **Projet 5** du parcours *Data Scientist* OpenClassrooms.

L’objectif :
> Exposer sous forme d’API un modèle de machine learning (classification du **risque de départ des salariés**), avec :
> - une **API FastAPI**,
> - une **base de données** pour logger les prédictions et ingérer le dataset d'entraînement (train),
> - un **pipeline CI/CD** (tests automatiques, déploiement),
> - une **documentation** claire pour l’installation et l’utilisation.

Le modèle de ML est issu du **Projet 4 : Classifiez automatiquement des informations** (modèle de turnover) et a été figé puis réutilisé ici.

### Prérequis

- Python : 3.13 (compatible avec la configuration CI)

- Poetry pour la gestion des dépendances

- Docker + docker compose (recommandé pour PostgreSQL local)

- Un accès internet pour installer les paquets nécessaires

---

## 1. Architecture du projet

Arborescence simplifiée :

```text
.
├── app/
│   ├── main.py            # API FastAPI (endpoints /predict, /predict_batch, /predict_csv)
│   ├── deps.py            # Chargement du modèle et des métadonnées
│   └── schemas.py         # Schémas Pydantic des requêtes/réponses
├── db/
│   ├── database.py        # Connexion SQLAlchemy (DATABASE_URL, SessionLocal)
│   ├── models.py          # Modèles de données (PredInput, PredOutput, DatasetFile, DatasetRow)
│   ├── crud.py            # Fonctions de logging des prédictions
│   └── create_db.py       # Script de création des tables
├── scripts/
│   ├── ingest_dataset.py  # Ingestion du dataset d'entraînement dans la BDD
│   ├── db_check.py        # Script de sanity-check de la base
│   └── seed_db.py         # (optionnel) Seed d’exemples
├── ml/
│   ├── model.joblib       # Modèle ML figé (issu du Projet 4)
│   └── model_meta.json    # Métadonnées : threshold, feature_names, etc.
├── data/
│   ├── train.csv          # Dataset d'entraînement (train, source du modèle)
│   └── README-data.md     # Notes spécifiques au dataset
├── tests/
│   ├── conftest.py        # Fixture FastAPI + FakeModel pour tests déterministes
│   ├── test_api_smoke.py  # Tests de santé /health et /meta
│   └── test_api_predict.py# Tests /predict, /predict_batch, /predict_csv (+ export CSV)
├── .github/
│   └── workflows/
│       ├── ci.yml         # Pipeline CI (tests + coverage)
│       └── cd.yml         # Pipeline CD (déploiement – ex. Hugging Face Space)
├── docker-compose.yml     # (optionnel) PostgreSQL local
├── Dockerfile             # Image de l’API
├── pyproject.toml         # Configuration Poetry
├── requirements.txt       # Dépendances runtime (alternative à Poetry)
├── requirements-dev.txt   # Dépendances dev (tests, linters…)
├── .env.sample            # Exemple de configuration (.env)
└── README.md              # Ce fichier

```

## 2. Configuration de l’environnement

### Cloner le dépôt

``` bash
git clone https://github.com/Thomas-Auvin/Projet-5.git
cd Projet-5

```

### Installer Poetry (si nécessaire)
Voir la documentation officielle de Poetry pour l’installation sur votre machine.

### Installer les dépendances

``` bash
poetry install

```

### Variables d’environnement

Copier le fichier d’exemple :

``` bash
cp .env.sample .env

```

Adapter les valeurs (connexion à la base, environnement, etc.) en suivant les indications présentes dans .env.sample.
👉 L’URL de base de données (par ex. `DATABASE_URL`) doit pointer vers :

- une base **PostgreSQL locale** (via Docker) pour le développement ; ou
- une base **SQLite** (ou autre) pour un déploiement type Hugging Face.

En l’état actuel, la configuration par défaut utilise une base **SQLite** (simple à déployer) via `DATABASE_URL`.
Il est cependant possible de basculer vers **PostgreSQL** en local via Docker (voir `docker-compose.yml`).

Important : ne jamais committer le .env.

## 3. Base de données & création du schéma

En local, vous pouvez utiliser PostgreSQL via Docker pour la base de données.

1. Lancer uniquement la base (optionnel)

Si vous souhaitez lancer PostgresSQL seul :

``` bash
docker compose up -d db

```

2. Créer la base et les tables

Un script Python gère la création de la base et des tables (modèles SQLAlchemy dans db/).

``` bash
poetry run python -m db.create_db

```


Ce script fait les choses suivantes :

1. crée la base si nécessaire, en fonction de `DATABASE_URL` ;
2. applique les modèles ORM, notamment :
   - `PredInput` / `PredOutput` : logs des prédictions (features d’entrée, proba, label, métadonnées, timestamps) ;
   - des tables associées au dataset d’entraînement (par ex. `dataset_files`, `dataset_rows`) pour historiser les données.


## 4. Lancer l’API en local (sans Docker)

Une fois les dépendances installées et la base prête, vous pouvez lancer l'application en local avec la commande suivante:

``` bash
poetry run uvicorn app.main:app --reload

```

Par défaut, une fois la commande lancée, l’API est disponible sur :

Swagger UI : http://localhost:8000/docs

OpenAPI JSON : http://localhost:8000/openapi.json

## 5. Pour lancer la stack complète (API + base) :

Si vous souhaitez lancer l'application en mode Docker, vous pouvez lancer la commande suivante :

``` bash
docker compose up --build

```

L’API écoute sur le port exposé dans docker-compose.yml (en général 8000).

Accès à la documentation : http://localhost:8000/docs.

## 6. Endpoints principaux

L'API est composée de différents endpoints. Voici leur description :

### GET /meta
Obtient les données utilisées par le modèle.

Exemple de réponse :

```json
{
  "version": "0.1.0",
  "threshold": 0.148,
  "model_meta": {
    "created_at": "2025-11-14T23:05:59.647624Z",
    "python": "3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]",
    "threshold": 0.148,
    "target": "a_quitte_l_entreprise",
    "feature_names": [
      "satisfaction_employee_environnement",
      "note_evaluation_precedente"
    ],
    "calibrated": true,
    "calibration_method": "isotonic",
    "random_state": 42,
    "note": "Hyperparamètres XGB figés dans ml/pipeline.py (issus du GridSearch P4)."
  }
}
```

### GET /health

Vérifie l’état de l’API (et éventuellement de la base / du modèle).

Exemple de réponse :

```json
{
  "status": "ok",
  "detail": "API running",
  "model_version": "1.0.0"
}
```

### POST /predict

Prédiction pour un seul salarié.

Body : un objet JSON correspondant au schéma d’entrée (features RH du salarié).
Le schéma exact est visible dans Swagger (/docs) via les modèles Pydantic.

Exemple simplifié de payload (à adapter aux vraies features du modèle) :

```json
{
  "age": 35,
  "departement": "Sales",
  "anciennete_annees": 4,
  "salaire": 42000,
  "heures_sup_moyennes": 5,
  "remote_ratio": 0.5
}
```

Exemple de réponse (structure indicative) :

```json
{
  "proba": 0.73,
  "label": 1
}
```

proba : probabilité que le salarié quitte l’entreprise (selon le modèle).

label : 1 = risque de départ, 0 = reste, après application du seuil métier (par ex. 0,148 issu du Projet 4).

### POST /predict_batch

Prédictions pour un lot de salariés.

Body : une liste d’objets d’entrée (schéma identique à /predict).

Réponse : une liste d’objets contenant proba et label pour chaque individu.

Exemple (structure indicative) :

```json
[
  { "proba": 0.73, "label": 1 },
  { "proba": 0.21, "label": 0 },
  { "proba": 0.58, "label": 1 }
]
```

Les schémas exacts (entrée/sortie) sont documentés dans /docs

### POST /predict_csv

Permet à l'utilisateur d'importer un CSV et de ressortir les prédictions du modèle en CSV.
Il est nécessaire que l'utilisateur indique le paramètre as_csv = true pour obtenir la sortie CSV.

## 7. Logging des prédictions

À chaque appel de /predict, /predict_batch ou predict_csv :

Les entrées sont sérialisées dans une table de type PredInput (features brutes, horodatage, éventuellement source de la requête).

Les sorties du modèle sont enregistrées dans une table PredOutput (proba, label, seuil utilisé, version du modèle, horodatage).

Un lien (FK) permet de rattacher une sortie à son entrée.

Cela permet :

d’analyser a posteriori les usages de l’API (qui est prédictible / non prédictible) ;

de rejouer des scénarios si le modèle évolue ;

de construire des dashboards métier (taux de scoring, profils à risque, etc.).

## 8. Ingestion du dataset d’entraînement

Dans le cadre du projet, le dataset d’entraînement est géré en 3 étapes :

1. ingestion du fichier d’entraînement (par ex. `data/train.csv`) dans la base ;
2. création d’une structure de tables de type :
   - `dataset_files` : métadonnées sur les fichiers (nom, date d’ingestion, hash, etc.) ;
   - `dataset_rows` : lignes de données associées à un fichier (features + label) ;
3. exécution d’un script d’ingestion dans le dossier `scripts/` (voir le code pour le détail exact).

L’exécution de l’ingestion du dataset d’entraînement se fait avec la commande suivante :

```bash
poetry run python -m scripts.ingest_dataset

```

## 9. Tests & qualité

Les tests sont regroupés dans le dossier tests/.

Tests unitaires pour les fonctions de base (chargement du modèle, logique métier, etc.).

Tests d’intégration / "smoke tests" pour l’API (ex. /health, /predict, /predict_batch, comportement en erreur, etc.).

Lancez tous les tests via la commande suivante:

```bash
poetry run pytest

```

Lancez les tests avec la couverture via la commande suivante :

``` bash
poetry run pytest --cov

```

Le rapport de couverture peut être exporté dans coverage.xml (utile pour la CI).

## 10. Intégration continue (GitHub Actions)

Un workflow CI (dans .github/workflows/) est intégré au projet. Le fonctionnement de la CI est le suivant:

1. installe le projet avec Poetry (Python 3.13) ;

2. exécute les tests (pytest) ;

    - vérifie la couverture ;

    - le linting (ruff, black) ;

    - tests de sécurité (bandit) ;

    - des checks spécifiques (par ex. démarrage de l’API avec une base temporaire).

La CI se déclenche sur :

    - les push sur les branches principales (ex. main) ;

    - les pull requests.

Un workflow de **CD** automatise le déploiement vers **Hugging Face Spaces**.
Chaque pull request pousse l’ensemble des modifications, qui deviennent accessibles via :

- le Space Hugging Face : https://huggingface.co/spaces/Thomas-Auvin/projet5-turnover-api

## 11. Déploiement sur Hugging Face Spaces

Le dépôt est configuré pour un déploiement sur Hugging Face Spaces en mode Docker :

Le front-matter en tête de ce README décrit la configuration du Space (sdk: docker).

Dockerfile construit l’image de l’API.

start.sh définit la commande de démarrage (lancement du serveur Uvicorn, éventuellement migrations / création de DB si SQLite).

Étapes typiques

Créer un nouveau Space sur Hugging Face :

Type : Docker.

Connecter le repo GitHub ou pousser le code sur le Space.

Configurer les variables d’environnement dans l’interface Hugging Face (similaires à celles du .env local, mais sans le commiter).

Laisser Hugging Face builder l’image Docker à partir de Dockerfile.

L’API sera alors accessible via l’URL publique du Space (https://huggingface.co/spaces/Thomas-Auvin/projet5-turnover-api), avec la même structure d’endpoints (/health, /predict, /predict_batch, /predict_csv).

## 12. Rappel sur le modèle de ML (Projet 4)

Le modèle embarqué dans cette API provient du Projet 4 "Classifiez automatiquement des informations" :

Modèle de type XGBoostClassifier encapsulé dans un pipeline scikit-learn :

préprocessing numérique (imputation médiane + standardisation) ;

encodage des variables catégorielles (OneHotEncoder) ;

gestion du déséquilibre via les hyperparamètres du modèle.

Métriques de référence (sur le P4) :

bonne séparation par rapport à une baseline "dummy" ;

choix d’un seuil métier (~0,148) pour privilégier le rappel (détection des départs) tout en gardant une précision acceptable.

Les artefacts de modèle (pipeline + métadonnées) sont sérialisés dans ml/ (par ex. model.joblib, model_meta.json) et chargés par l’API au démarrage.

## 13. Auteur

Thomas Auvin
Data Scientist en formation (OpenClassrooms)

Ce projet est réalisé dans le cadre du Projet 5 – Industrialisez un modèle de machine learning.
