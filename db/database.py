# db/database.py
from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Charge .env à la racine (DATABASE_URL, etc.)
load_dotenv()

# Exemple .env pour PostgreSQL local :
# DATABASE_URL=postgresql+psycopg://postgres:VOTRE_MDP@localhost:5432/projet5
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local.db")

# echo=False (mets True pour voir les requêtes SQL)
engine = create_engine(DATABASE_URL, echo=False, future=True)

# Session factory SQLAlchemy v2
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
    expire_on_commit=False,   # <-- ajout clé
)


def get_db():
    """
    Dépendance FastAPI : yield une session puis la ferme.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
