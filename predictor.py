"""
predictor.py — PredictIQ Pro v5.0
Lazy imports to prevent Railway startup crash.
"""

import math
import os
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = os.getenv("MODEL_PATH", "models/ensemble_v5.pkl")
_model = None


def get_model():
    global _model
    if _model is None and Path(MODEL_PATH).exists():
        import joblib
        _model = joblib.load(MODEL_PATH)
    return _model


# ── Poisson helpers ────────────────────────────────────

def _poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def _score_matrix(home_xg: float, away_xg: float, max_g: int = 6) -> dict:
    return {
        (h, a): _poisson(home_xg, h) * _poisson(away_xg, a)
        for h in range(max_g + 1)
        for a in range(max_g + 1)
    }

def most_likely_score(home_xg: float, away_xg: float) -> str:
    sm = _score_matrix(home_xg, away_xg)
    h, a = max(sm, key=sm.get)
    return f"{h}-{a}"

def over_under_probs(home_xg: float, away_xg: float):
    sm = _score_matrix(home_xg, away_xg)
    p15  = round(sum(p for (h,a),p in sm.items() if h+a > 1) * 100, 1)
    p25  = round(sum(p for (h,a),p in sm.items() if h+a > 2) * 100, 1)
    p35  = round(sum(p for (h,a),p in sm.items() if h+a > 3) * 100, 1)
    btts = round(sum(p for (h,a),p in sm.items() if h > 0 and a > 0) * 100, 1)
    return p15, p25, p35, btts


# ── Main predict function ──────────────────────────────

def predict(input_data: dict) -> dict:
    from features import engineer_features

    model = get_model()
    if model is None:
        raise RuntimeError("Model not loaded. Train and push models/ensemble_v5.pkl first.")

    df = pd.DataFrame([input_data])
    X  = engineer_features(df)

    league_id = int(input_data.get("league_id", 0))
    proba     = model.predict_proba(X, league_id=league_id)[0]
    p_home, p_draw, p_away = float(proba[0]), float(proba[1]), float(proba[2])
    predicted = ["H", "D", "A"][int(np.argmax(proba))]

    home_xg = float(input_data.get("bzz_expected_home_goals",
                    input_data.get("poisson_home_goals", 1.4)))
    away_xg = float(input_data.get("bzz_expected_away_goals",
                    input_data.get("poisson_away_goals", 1.1)))

    home_xg = home_xg * 0.7 + (p_home * 2.5) * 0.3
    away_xg = away_xg * 0.7 + (p_away * 2.5) * 0.3

    p15, p25, p35, p_btts = over_under_probs(home_xg, away_xg)
    confidence = min(float(max(proba)) * 1.05, 0.99)
    fav        = "H" if p_home > p_away else ("A" if p_away > p_home else None)
    fav_prob   = max(p_home, p_away) * 100

    return {
        "prob_home_win":    round(p_home * 100, 1),
        "prob_draw":        round(p_draw * 100, 1),
        "prob_away_win":    round(p_away * 100, 1),
        "predicted_result": predicted,
        "expected_home_goals": round(home_xg, 2),
        "expected_away_goals": round(away_xg, 2),
        "most_likely_score":   most_likely_score(home_xg, away_xg),
        "prob_over_15": p15,
        "prob_over_25": p25,
        "prob_over_35": p35,
        "prob_btts":    p_btts,
        "confidence":        round(confidence, 3),
        "model_version":     "Ensemble v5.0 (CB+XGB+LGB)",
        "features_used":     len(X.columns),
        "spatial_data_used": any(k.startswith("bzz_") for k in input_data),
        "favorite":           fav,
        "favorite_prob":      round(fav_prob, 1),
        "favorite_recommend": fav_prob >= 55.0,
        "over_15_recommend":  p15 >= 72.0,
        "over_25_recommend":  p25 >= 55.0,
        "over_35_recommend":  p35 >= 28.0,
        "btts_recommend":     p_btts >= 52.0,
        "winner_recommend":   predicted != "D" and max(p_home, p_away) >= 0.50,
    }
    
