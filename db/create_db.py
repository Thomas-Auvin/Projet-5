"""
Script pour créer les tables SQLAlchemy dans la base de données.

Hypothèses :
- La base existe déjà (Postgres ou SQLite).
- L'URL de connexion est gérée dans db.database
  (DATABASE_URL ou fallback local.db).
- Tous les modèles héritent de Base dans db.models.
"""

from db.database import engine  # engine basé sur DATABASE_URL + .env
from db.models import Base  # Base + enregistrement des modèles


def init_db() -> None:
    """Crée toutes les tables définies dans Base.metadata."""
    Base.metadata.create_all(bind=engine)
    print("✅ Tables créées (ou déjà existantes).")


if __name__ == "__main__":
    init_db()
