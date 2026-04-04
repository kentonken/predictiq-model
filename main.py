import os
import pandas as pd
import numpy as np
from typing import List, Union, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostClassifier

app = FastAPI()

# 1. Load the CatBoost model
MODEL_PATH = "catboost_model.cbm"
model = CatBoostClassifier()

if os.path.exists(MODEL_PATH):
    try:
        model.load_model(MODEL_PATH)
        MODEL_LOADED = True
    except Exception as e:
        print(f"Error loading model: {e}")
        MODEL_LOADED = False
else:
    MODEL_LOADED = False

# 2. Updated Schema with Optional fields to prevent 422 errors
class MatchData(BaseModel):
    home_team: str
    away_team: str
    # Making these optional with defaults ensures the API doesn't crash if Bzzoiro misses a field
    home_league_position: Optional[int] = 0
    away_league_position: Optional[int] = 0
    home_pts: Optional[int] = 0
    away_pts: Optional[int] = 0
    home_possession: Optional[float] = 50.0
    away_possession: Optional[float] = 50.0
    home_shots: Optional[int] = 0
    away_shots: Optional[int] = 0
    home_avg_goals_scored: Optional[float] = 0.0
    away_avg_goals_conceded: Optional[float] = 0.0
    league_avg_goals: Optional[float] = 1.35

def calculate_poisson_lambda(scored, conceded, avg):
    if avg == 0: return 0.0
    # Your hybrid formula: (scored/avg) * (conceded/avg) * avg
    return (scored / avg) * (conceded / avg) * avg

@app.get("/health")
async def health():
    return {
        "status": "online",
        "model_loaded": MODEL_LOADED
    }

@app.post("/predict_hybrid")
async def predict_hybrid(data: Union[MatchData, List[MatchData]]):
    if not MODEL_LOADED:
        return {"error": "Model not loaded on server."}

    # Handle both single objects and lists
    if not isinstance(data, list):
        matches = [data]
    else:
        matches = data

    try:
        rows = []
        matches_info = []

        for m in matches:
            h_lambda = calculate_poisson_lambda(
                m.home_avg_goals_scored,
                m.away_avg_goals_conceded,
                m.league_avg_goals
            )

            # This must exactly match the feature order your model was trained on
            rows.append({
                "home_league_position": m.home_league_position,
                "away_league_position": m.away_league_position,
                "home_pts": m.home_pts,
                "away_pts": m.away_pts,
                "home_possession": m.home_possession,
                "away_possession": m.away_possession,
                "home_shots": m.home_shots,
                "away_shots": m.away_shots,
                "poisson_home_lambda": h_lambda
            })
            
            matches_info.append(f"{m.home_team} vs {m.away_team}")

        df = pd.DataFrame(rows)

        # Get probabilities for class 1 (usually 'Over 2.5' or 'Win')
        probs = model.predict_proba(df)[:, 1]

        final_predictions = []
        for i, prob in enumerate(probs):
            # Market Decision Logic
            prediction = "OVER 2.5" if prob > 0.55 else "UNDER 2.5"

            # Value Bet Detection
            current_avg = matches[i].league_avg_goals
            is_value = bool(current_avg > 3.0 and prob > 0.6)

            final_predictions.append({
                "match": matches_info[i],
                "market": "Goals (O/U 2.5)",
                "prediction": prediction,
                "probability": f"{round(float(prob) * 100, 1)}%",
                "is_value_bet": is_value
            })

        return {"predictions": final_predictions}

    except Exception as e:
        return {"error": str(e)}
        
