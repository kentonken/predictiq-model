import os
import pandas as pd
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostClassifier
import numpy as np

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

class MatchData(BaseModel):
    home_team: str
    away_team: str
    home_league_position: int
    away_league_position: int
    home_pts: int
    away_pts: int
    home_possession: float
    away_possession: float
    home_shots: int
    away_shots: int
    home_avg_goals_scored: float
    away_avg_goals_conceded: float
    league_avg_goals: float = 1.35

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
async def predict_hybrid(data: List[MatchData]):
    if not MODEL_LOADED:
        return {"error": "Model not loaded on server."}
    
    try:
        rows = []
        matches_info = []
        
        for m in data:
            h_lambda = calculate_poisson_lambda(
                m.home_avg_goals_scored, 
                m.away_avg_goals_conceded, 
                m.league_avg_goals
            )
            
            # This must match the features your model was trained on
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
        # Get probabilities for class 1 (usually the 'Over' or 'Win' class)
        probs = model.predict_proba(df)[:, 1] 
        
        final_predictions = []
        for i, prob in enumerate(probs):
            # Market Decision Logic
            prediction = "OVER 2.5" if prob > 0.55 else "UNDER 2.5"
            
            # Value Bet Detection: High League Avg + High ML Probability
            is_value = (data[i].league_avg_goals > 3.0) and (prob > 0.6)
            
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
        
