import os
import numpy as np
from catboost import CatBoostClassifier
from features import FEATURE_COLUMNS

MODEL_PATH = os.getenv("MODEL_PATH", "catboost_model.cbm")
OUTCOME_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}

def load_model():
    model = CatBoostClassifier()
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
        print(f"CatBoost model loaded from {MODEL_PATH}")
    else:
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run train.py locally first, then commit model/catboost_model.cbm."
        )
    return model

def predict_match(model, features: dict, fixture_id: int) -> dict:
    feature_vector = [features.get(col, 0) for col in FEATURE_COLUMNS]
    X = np.array([feature_vector])
    probs = model.predict_proba(X)[0]
    predicted_class = int(np.argmax(probs))
    prediction_label = OUTCOME_LABELS[predicted_class]
    confidence = float(probs[predicted_class])
    return {
        "fixture_id": fixture_id,
        "home_team": features.get("home_team", ""),
        "away_team": features.get("away_team", ""),
        "prediction": prediction_label,
        "confidence": round(confidence, 4),
        "home_win_prob": round(float(probs[0]), 4),
        "draw_prob": round(float(probs[1]), 4),
        "away_win_prob": round(float(probs[2]), 4),
        "tip": _build_tip(prediction_label, confidence),
    }

def _build_tip(prediction: str, confidence: float) -> str:
    pct = round(confidence * 100)
    if pct >= 70: strength = "Strong Pick"
    elif pct >= 55: strength = "Good Pick"
    else: strength = "Lean"
    return f"{strength}: {prediction} ({pct}%
