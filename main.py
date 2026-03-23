import os
import pandas as pd
from typing import List
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

class MatchData(BaseModel):
    home_team: str
    away_team: str
    home_league_position: int
    away_league_position: int
    home_pts: int
    away_pts: int
    home_possession: float
    away_possession: float
    home_shots: float
    away_shots: float
    home_avg_goals_scored: float
    away_avg_goals_conceded: float
    league_avg_goals: float = 1.35 

def calculate_poisson_lambda(scored, conceded, avg):
    if avg == 0: return 0.0
    return (scored / avg) * (conceded / avg) * avg

@app.get("/")
async def health():
    return {
        "status": "online",
        "model_file_exists": os.path.exists(MODEL_PATH),
        "model_successfully_loaded": MODEL_LOADED
    }

@app.post("/predict_hybrid")
async def predict_hybrid(data: List[MatchData]):
    if not MODEL_LOADED:
        return {"error": "Model not loaded on server. Check logs."}
    try:
        rows = []
        for m in data:
            h_lambda = calculate_poisson_lambda(
                m.home_avg_goals_scored, 
                m.away_avg_goals_conceded, 
                m.league_avg_goals
            )
            
            rows.append({
                "home_league_position": m.home_league_position,
                "away_league_position": m.away_league_position,
                "home_pts": m.home_pts,
                "away_pts": m.away_pts,
                "home_possession": m.home_possession,
                "away_possession": m.away_possession,
                "home_shots": m.home_shots,
                "away_shots": m.away_shots,
                "poisson_home_lambda": h_lambda # Ensure this matches Colab exactly!
            })

        df = pd.DataFrame(rows)
        probs = model.predict_proba(df)
        return {"predictions": [float(p[1]) for p in probs]}
    except Exception as e:
        return {"error": str(e)}
        
