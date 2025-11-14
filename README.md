# Projet 5 — Turnover API (FastAPI + P4)

**But** : déployer le modèle de classification du Projet 4 via une API FastAPI, avec DB PostgreSQL, tests Pytest et CI/CD.

## Démarrer
```bash
# Installer les dépendances (Poetry)
poetry install

# Lancer l'API en local
poetry run uvicorn app.main:app --reload
