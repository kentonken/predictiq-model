from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from catboost import CatBoostClassifier
import os

app = FastAPI()

# Load the model you trained in Colab
MODEL_PATH = "catboost_model.cbm"
model = CatBoostClassifier()
if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)

class MatchData(BaseModel):
    home_team: str
    away_team: str
    # Stats from TheSportsDB
    home_league_position: int
    away_league_position: int
    home_pts: int
    away_pts: int
    # Live stats from Firecrawl (via Lovable)
    home_possession: float
    away_possession: float
    home_shots: float
    away_shots: float
    # Poisson inputs
    home_avg_goals_scored: float
    away_avg_goals_conceded: float
    league_avg_goals: float = 1.35 

def calculate_poisson_lambda(scored, conceded, avg):
    """Calculates the Poisson home win probability factor."""
    if avg == 0: return 0.0
    return (scored / avg) * (conceded / avg) * avg

@app.post("/predict/hybrid")
async def predict_hybrid(data: List[MatchData]):
    try:
        rows = []
        for m in data:
            # Generate the Poisson feature your model needs
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
                "poisson_home_lambda": h_lambda # Feature name must match your training set
            })

        df = pd.DataFrame(rows)
        # Your model's prediction probability for a Home Win (Index 1)
        probs = model.predict_proba(df)
        return {"predictions": [float(p[1]) for p in probs]}
        
    except Exception as e:
        return {"error": str(e)}
        
