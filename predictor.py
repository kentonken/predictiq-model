import os
import joblib
import logging
from pathlib import Path

# Setup logging and paths
logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/ensemble_v5.pkl"))
_model = None

def get_model():
    """Returns the loaded v5 ensemble model."""
    global _model
    if _model is not None:
        return _model
    
    if MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
            logger.info("✅ Ensemble v5 loaded successfully.")
            return _model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    logger.warning("⚠️ No model found locally.")
    return None

# Keep your existing predict(input_data) function below this...
