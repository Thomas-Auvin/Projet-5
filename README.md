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
> - une **base de données** pour logger les prédictions et ingérer le dataset de train,
> - une **pipeline CI/CD** (tests automatiques, déploiement),
> - une **documentation** claire pour l’installation et l’utilisation.

Le modèle de ML est issu du **Projet 4** (modèle de turnover) et a été figé puis réutilisé ici.

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
│   ├── models.py          # Modèle de données (PredInput, PredOutput, DatasetFile, DatasetRow)
│   ├── crud.py            # Fonctions de logging des predictions
│   └── create_db.py       # Script de création des tables
├── scripts/
│   ├── ingest_dataset.py  # Ingestion du dataset de train dans la BDD
│   ├── db_check.py        # Script de sanity-check de la base
│   └── seed_db.py         # (optionnel) Seed d’exemples
├── ml/
│   ├── model.joblib       # Modèle ML figé (issu du Projet 4)
│   └── model_meta.json    # Métadonnées : threshold, feature_names, etc.
├── data/
│   ├── train.csv          # Dataset de train (source du modèle)
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

Cloner le dépôt

git clone https://github.com/Thomas-Auvin/Projet-5.git
cd Projet-5


Installer Poetry (si nécessaire)
Voir la doc officielle Poetry pour l’installation sur ta machine.

Installer les dépendances

poetry install


Variables d’environnement

Copier le fichier d’exemple :

cp .env.sample .env


Adapter les valeurs (connexion à la base, environnement, etc.) en suivant les indications présentes dans .env.sample.
👉 L’URL de base de données (par ex. DATABASE_URL) doit pointer vers :

un PostgreSQL local (via Docker) pour le dev, ou

une base SQLite ou autre pour un déploiement type Hugging Face.

Important : ne jamais committer le .env.

## 3. Base de données & création du schéma

En local, la base est gérée via PostgreSQL dans Docker.

1. Lancer uniquement la base (optionnel)

Si tu veux lancer Postgres seul :

docker compose up -d db

2. Créer la base et les tables

Un script Python gère la création de la base et des tables (modèles SQLAlchemy dans db/).

poetry run python -m db.create_db


Ce script :

crée la base si nécessaire ;

applique les modèles ORM :

PredInput / PredOutput : logs des prédictions (features d’entrée, proba, label, métadonnées, timestamps) ;

(Option B) éventuellement des tables associées au dataset d’entraînement (par ex. dataset_files, dataset_rows) pour historiser les données.

## 4. Lancer l’API en local (sans Docker)

Une fois les dépendances installées et la base prête :

poetry run uvicorn app.main:app --reload


Par défaut, l’API est disponible sur :

Swagger UI : http://localhost:8000/docs

OpenAPI JSON : http://localhost:8000/openapi.json

## 5. Pour lancer la stack complète (API + base) :

docker compose up --build


L’API écoute sur le port exposé dans docker-compose.yml (en général 8000).

Accès à la documentation : http://localhost:8000/docs.

## 6. Endpoints principaux
GET /health

Vérifie l’état de l’API (et éventuellement de la base / du modèle).

Exemple de réponse :

{
  "status": "ok",
  "detail": "API running",
  "model_version": "1.0.0"
}

POST /predict

Prédiction pour un seul salarié.

Body : un objet JSON correspondant au schéma d’entrée (features RH du salarié).
Le schéma exact est visible dans Swagger (/docs) via les modèles Pydantic.

Exemple simplifié de payload (à adapter aux vraies features du modèle) :

{
  "age": 35,
  "departement": "Sales",
  "anciennete_annees": 4,
  "salaire": 42000,
  "heures_sup_moyennes": 5,
  "remote_ratio": 0.5
}


Exemple de réponse (structure indicative) :

{
  "proba": 0.73,
  "label": 1
}


proba : probabilité que le salarié quitte l’entreprise (selon le modèle).

label : 1 = risque de départ, 0 = reste, après application du seuil métier (par ex. 0,148 issu du Projet 4).

POST /predict_batch

Prédictions pour un lot de salariés.

Body : une liste d’objets d’entrée (schéma identique à /predict).

Réponse : une liste d’objets contenant proba et label pour chaque individu.

Exemple (structure indicative) :

[
  { "proba": 0.73, "label": 1 },
  { "proba": 0.21, "label": 0 },
  { "proba": 0.58, "label": 1 }
]


Les schémas exacts (entrée/sortie) sont documentés dans /docs

## 7. Logging des prédictions

À chaque appel de /predict ou /predict_batch :

Les entrées sont sérialisées dans une table de type PredInput (features brutes, horodatage, éventuellement source de la requête).

Les sorties du modèle sont enregistrées dans une table PredOutput (proba, label, seuil utilisé, version du modèle, horodatage).

Un lien (FK) permet de rattacher une sortie à son entrée.

Cela permet :

d’analyser a posteriori les usages de l’API (qui est prédictible / non prédictible) ;

de rejouer des scénarios si le modèle évolue ;

de construire des dashboards métier (taux de scoring, profils à risque, etc.).

## 8. Ingestion du dataset d’entraînement

Dans le cadre de l’option B du sujet, le projet prévoit :

l’ingestion du fichier d’entraînement (par ex. data/train.csv) dans la base ;

une structure de tables de type :

dataset_files : métadonnées sur les fichiers (nom, date d’ingestion, hash, etc.) ;

dataset_rows : lignes de données associées à un fichier (features + label).

Un script d’ingestion est présent dans le dossier scripts/ (voir le code pour le nom et l’usage exact).
L’idée générale :

poetry run python -m scripts.ingest_dataset  # exemple de commande

## 9. Tests & qualité

Les tests sont regroupés dans le dossier tests/.

Tests unitaires pour les fonctions de base (chargement du modèle, logique métier, etc.).

Tests d’intégration / "smoke tests" pour l’API (ex. /health, /predict, /predict_batch, comportement en erreur, etc.).

Lancer tous les tests :

poetry run pytest


Avec la couverture :

poetry run pytest --cov


Le rapport de couverture peut être exporté dans coverage.xml (utile pour la CI).

## 10. Intégration continue (GitHub Actions)

Un workflow CI (dans .github/workflows/) :

installe le projet avec Poetry (Python 3.13) ;

exécute les tests (pytest) ;

vérifie la couverture ;

peut être étendu pour :

le linting (flake8, black, isort, etc.) ;

des checks spécifiques (par ex. démarrage de l’API avec une base temporaire).

La CI se déclenche sur :

les push sur les branches principales (ex. main) ;

les pull requests.

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

L’API sera alors accessible via l’URL publique du Space, avec la même structure d’endpoints (/health, /predict, /predict_batch).

## 12. Rappel sur le modèle de ML (Projet 4)

Le modèle embarqué dans cette API provient du Projet 4 :

Modèle de type XGBoost classifier encapsulé dans un pipeline scikit-learn :

préprocessing numérique (imputation médiane + standardisation) ;

encodage des variables catégorielles (OneHotEncoder) ;

gestion du déséquilibre via les hyperparamètres du modèle.

Métriques de référence (sur le P4) :

bonne séparation par rapport à une baseline "dummy" ;

choix d’un seuil métier (~0,148) pour privilégier le rappel (détection des départs) tout en gardant une précision acceptable.

Les artefacts de modèle (pipeline + métadonnées) sont sérialisés dans ml/ (par ex. model.joblib, model_meta.json) et chargés par l’API au démarrage.

## 13. Auteur

Thomas Auvin
Docteur en psychologie sociale & Data Scientist en formation (OpenClassrooms)

Ce projet est réalisé dans le cadre du Projet 5 – Industrialisez un modèle de machine learning.
