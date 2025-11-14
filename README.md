# Projet 5 — Turnover API (FastAPI + P4)

**But** : déployer le modèle de classification du Projet 4 via une API FastAPI, avec DB PostgreSQL, tests Pytest et CI/CD.

## Démarrer
```bash
# Installer les dépendances (Poetry)
poetry install

# Lancer l'API en local
poetry run uvicorn app.main:app --reload

# Comment lancer les tests 
poetry run pytest -q

# Comment lancer l'API
poetry run python -m uvicorn app.main:app --reload
    # Puis ouvre http://127.0.0.1:8000/docs
