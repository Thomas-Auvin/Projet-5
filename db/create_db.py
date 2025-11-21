# db/create_db.py
from __future__ import annotations
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local.db")


def ensure_database_exists():
    """
    Si DATABASE_URL pointe vers PostgreSQL, se connecte à la DB 'postgres'
    pour créer la base cible si elle n'existe pas encore. (No-op pour SQLite.)
    """
    if not DATABASE_URL.startswith("postgresql+psycopg://"):
        return  # SQLite (ou autre) : rien à faire

    url = make_url(DATABASE_URL)
    dbname = url.database
    admin_url = url.set(database="postgres")  # DB d'admin

    admin_engine = create_engine(
        admin_url, isolation_level="AUTOCOMMIT", future=True
        )
    with admin_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": dbname},
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{dbname}"'))
    admin_engine.dispose()


def create_tables():
    from db.models import Base
    engine = create_engine(DATABASE_URL, future=True)
    Base.metadata.create_all(bind=engine)


def init_db():
    ensure_database_exists()
    create_tables()


if __name__ == "__main__":
    init_db()
    print("✅ Tables créées/ok")
