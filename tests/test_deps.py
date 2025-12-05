# tests/test_deps.py
from app import deps


def test_load_meta_ok():
    """Vérifie que le meta du modèle est bien chargé depuis ml/model_meta.json."""
    meta = deps.load_meta()
    # On ne fige pas la structure exacte, mais on teste quelques clés importantes.
    assert isinstance(meta, dict)
    assert "threshold" in meta or "default_threshold" in meta
    assert "feature_names" in meta
