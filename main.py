from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from catboost import CatBoostClassifier
import requests
import os

app = FastAPI()

# Load the CatBoost model
model = CatBoostClassifier().load_model("catboost_model.cbm")

class MatchData(BaseModel):
    # From TheSportsDB / Lovable State
    home_team: str
    away_team: str
    home_league_position: int
    away_league_position: int
    home_pts: int
    away_pts: int
    # From Firecrawl (Scraped via Lovable)
    home_possession: float
    away_possession: float
    home_shots: float
    away_shots: float
    # Historical Goals (for Poisson)
    home_avg_goals_scored: float
    away_avg_goals_conceded: float
    league_avg_goals: float = 1.35 # Default for most leagues

def calculate_poisson_lambda(scored, conceded, avg):
    # Calculates the 'Expected Goals' feature
    if avg == 0: return 0.0
    return (scored / avg) * (conceded / avg) * avg

@app.post("/predict/hybrid")
async def predict_hybrid(data: List[MatchData]):
    try:
        input_list = []
        for match in data:
            # Calculate Poisson Expected Goals
            home_exp_g = calculate_poisson_lambda(
                match.home_avg_goals_scored, 
                match.away_avg_goals_conceded, 
                match.league_avg_goals
            )
            
            # Map everything to the columns your model expects
            row = {
                "home_league_position": match.home_league_position,
                "away_league_position": match.away_league_position,
                "home_pts": match.home_pts,
                "away_pts": match.away_pts,
                "home_possession": match.home_possession,
                "away_possession": match.away_possession,
                "home_shots": match.home_shots,
                "away_shots": match.away_shots,
                "poisson_home_lambda": home_exp_g # New 'Skill-Stacked' feature
            }
            input_list.append(row)

        df = pd.DataFrame(input_list)
        probs = model.predict_proba(df)
        
        return {"predictions": [float(p[1]) for p in probs]}
    except Exception as e:
        return {"error": str(e)}
        
