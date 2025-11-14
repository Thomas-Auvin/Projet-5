from fastapi import FastAPI

app = FastAPI(
    title="Futurisys Turnover API",
    description="POC Projet 5 : API FastAPI pour le modèle du Projet 4",
    version="0.1.0",
)

@app.get("/health")
def health_check():
    return {"status": "ok"}
