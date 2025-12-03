# Dockerfile
FROM python:3.13-slim

# --- OS deps (xgboost a besoin de libgomp) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# --- Env de base ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Installer les deps Python via requirements.txt ---
# (copié seul d’abord pour profiter du cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copier le code du projet ---
COPY app app
COPY db db
COPY ml ml
COPY scripts scripts
COPY src src
COPY README.md README.md
COPY start.sh start.sh

# --- Droits d’exécution du script de démarrage ---
RUN chmod +x start.sh

# --- Variables d'env utiles ---
ENV MODEL_PATH="ml/model.joblib"
ENV MODEL_META_PATH="ml/model_meta.json"
ENV PORT=7860

# --- Port d’écoute (HF fournira PORT) ---
EXPOSE 7860

# --- Lancement ---
CMD ["bash", "start.sh"]
