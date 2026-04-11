import joblib
import os
from pathlib import Path

# Global variable to cache the model
_model = None
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/ensemble_v5.pkl"))

def get_model():
    """Function to load and return the model ensemble."""
    global _model
    if _model is not None:
        return _model
    
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        return _model
    
    # Fallback if model doesn't exist yet
    return None

def predict(input_data: dict) -> dict:
    """Main prediction logic engine."""
    model = get_model()
    if model is None:
        raise RuntimeError("No model loaded. Call /train first.")
    
    # ... rest of your existing prediction code ...
    
