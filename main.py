from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI
import pandas as pd
from catboost import CatBoostClassifier
import os

app = FastAPI()

MODEL_PATH = "catboost_model.cbm"
model = CatBoostClassifier()

if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)
else:
    print("Model file missing!")

class MatchInput(BaseModel):
    # Basic info
    home_team: str
    away_team: str
    
    # The stats your model is looking for (based on your training error)
    home_league_position: int = 0
    away_league_position: int = 0
    home_pts: int = 0
    away_pts: int = 0
    home_avg_goals: float = 0.0
    away_avg_goals: float = 0.0
    home_h2h_wins: int = 0
    away_h2h_wins: int = 0

@app.post("/predict/batch")
async def predict_batch(data: List[MatchInput]):
    try:
        # 1. Convert to DataFrame
        df = pd.DataFrame([d.dict() for d in data])
        
        # 2. DROP only the name columns
        # The model needs the rest to match its 'pool'
        features = df.drop(columns=['home_team', 'away_team'])
        
        # 3. Predict
        probs = model.predict_proba(features)
        
        # 4. Return Home Win probability (Index 1)
        results = [float(p[1]) for p in probs]
        return {"predictions": results}
    
    except Exception as e:
        return {"error": str(e)}
        
