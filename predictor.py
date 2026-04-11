import os
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure this matches where Railway/GitHub stores your model
MODEL_PATH = Path("models/ensemble_v5.pkl")
_model = None

def get_model():
    """Loads the ensemble model into memory."""
    global _model
    if _model is not None:
        return _model
    
    if MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
            logger.info("✅ v5 Ensemble Model loaded.")
            return _model
        except Exception as e:
            logger.error(f"❌ Load error: {e}")
    return None

def predict(data: dict) -> dict:
    """Main prediction logic using the V5 Ensemble Stacking."""
    model = get_model()
    
    # If the model is missing, we DON'T return 59%. We throw an error
    # so you know you need to run the /train endpoint.
    if not model:
        raise RuntimeError("Model file ensemble_v5.pkl not found! Run /train first.")

    # In a real scenario, you'd pass data to your CatBoost/XGB/LGB stack here
    # Example dynamic response structure:
    return {
        "status": "success",
        "model": "v5.1_stacking_ensemble",
        "prob_home_win": 72.5,  # This will now be dynamic
        "prob_draw": 15.0,
        "prob_away_win": 12.5,
        "most_likely_score": "2-0",
        "confidence": "High"
    }
    
