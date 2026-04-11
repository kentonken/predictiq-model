import numpy as np
import pandas as pd
from features import engineer_features

def predict(input_data: dict) -> dict:
    model = get_model()
    
    # 1. Engineer features from the RAW input
    df = pd.DataFrame([input_data])
    X = engineer_features(df)
    
    # 2. Get Model Probabilities
    # If CatBoost sees real ELO diffs, these will finally change!
    proba = model.predict_proba(X)[0]
    p_home, p_draw, p_away = proba[0], proba[1], proba[2]

    # 3. Dynamic Poisson Logic
    # Instead of defaulting to 1.4, we calculate xG based on the Model Proba
    # This ensures a '59%' win chance results in a realistic score like 2-1 or 2-0
    home_xg = float(input_data.get("bzz_expected_home_goals") or (p_home * 2.5))
    away_xg = float(input_data.get("bzz_expected_away_goals") or (p_away * 2.5))

    # Calculate Score & O/U Probs using your Poisson helpers
    p15, p25, p35, btts = over_under_probs(home_xg, away_xg)
    score = most_likely_score(home_xg, away_xg)

    return {
        "prob_home_win": round(p_home * 100, 1),
        "prob_draw": round(p_draw * 100, 1),
        "prob_away_win": round(p_away * 100, 1),
        "most_likely_score": score,
        "over_25_prob": p25,
        "btts_prob": btts,
        "confidence": round(max(proba) * 100, 1)
    }
    
