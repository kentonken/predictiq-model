import os
import math
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from features import engineer_features

logger = logging.getLogger(__name__)

# Model Path Configuration
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/ensemble_v5.pkl"))
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model

    # 1. Try local file (Railway/GitHub)
    if MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
            logger.info("✅ Model loaded from local file")
            return _model
        except Exception as e:
            logger.error(f"❌ Failed to load local model: {e}")

    # 2. Fallback to Supabase download if local fails
    from model_store import download_model
    if download_model() and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded from Supabase Storage")
        return _model

    logger.warning("⚠️ No model available - predicting with neutral baseline")
    return None

# --- Poisson Probability Helpers ---

def _poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def _score_matrix(home_xg: float, away_xg: float, max_g: int = 6) -> dict:
    matrix = {}
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            matrix[(h, a)] = _poisson(home_xg, h) * _poisson(away_xg, a)
    return matrix

def most_likely_score(home_xg: float, away_xg: float) -> str:
    sm = _score_matrix(home_xg, away_xg)
    h, a = max(sm, key=sm.get)
    return f"{h}-{a}"

def over_under_probs(home_xg: float, away_xg: float):
    sm = _score_matrix(home_xg, away_xg)
    p15 = round(sum(p for (h, a), p in sm.items() if h + a > 1.5) * 100, 1)
    p25 = round(sum(p for (h, a), p in sm.items() if h + a > 2.5) * 100, 1)
    p35 = round(sum(p for (h, a), p in sm.items() if h + a > 3.5) * 100, 1)
    btts = round(sum(p for (h, a), p in sm.items() if h > 0 and a > 0) * 100, 1)
    return p15, p25, p35, btts

# --- Main Prediction Engine ---

def predict(input_data: dict) -> dict:
    model = get_model()
    
    # Create DataFrame and apply feature engineering
    df = pd.DataFrame([input_data])
    X = engineer_features(df)
    
    # Get ML Probabilities (The core of the prediction)
    if model:
        # Some ensembles require league_id for spatial context
        lid = int(input_data.get("league_id", 0))
        try:
            proba = model.predict_proba(X)[0]
        except:
            # Fallback if the ensemble requires league_id as a parameter
            proba = model.predict_proba(X, league_id=lid)[0]
            
        p_home, p_draw, p_away = proba[0], proba[1], proba[2]
    else:
        # Strict Neutral Fallback if model is totally missing
        p_home, p_draw, p_away = 0.33, 0.34, 0.33

    # Dynamic XG Calculation (Replacing the 1.4 hard-code)
    # We blend Bzzoiro data (if available) with ML model performance
    bzz_h = float(input_data.get("bzz_expected_home_goals", 1.2))
    bzz_a = float(input_data.get("bzz_expected_away_goals", 1.0))
    
    # Logic: Blend Bzzoiro XG (70%) with ML-derived XG (30%)
    # This ensures that if Al-Nassr is favored by ML, the score reflects it.
    home_xg = (bzz_h * 0.7) + (p_home * 3.0 * 0.3)
    away_xg = (bzz_a * 0.7) + (p_away * 3.0 * 0.3)

    # Get Score and O/U stats based on the dynamic XG
    p15, p25, p35, p_btts = over_under_probs(home_xg, away_xg)
    pred_score = most_likely_score(home_xg, away_xg)
    
    # Determine predicted result (H, D, A)
    results_map = ["H", "D", "A"]
    predicted = results_map[np.argmax([p_home, p_draw, p_away])]

    return {
        "prob_home_win": round(p_home * 100, 1),
        "prob_draw": round(p_draw * 100, 1),
        "prob_away_win": round(p_away * 100, 1),
        "predicted_result": predicted,
        "most_likely_score": pred_score,
        "over_15_prob": p15,
        "over_25_prob": p25,
        "over_35_prob": p35,
        "btts_prob": p_btts,
        "confidence": round(max(p_home, p_away) * 100, 1),
        "model_version": "Ensemble v5.0 (CB+XGB+LGB)",
        "recommendation": "High" if max(p_home, p_away) > 0.65 else "Standard"
        }
    
