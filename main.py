from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI

app = FastAPI()

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_possession: float = 50.0
    away_possession: float = 50.0
    home_shots: float = 0.0
    away_shots: float = 0.0
    avg_goals_last_5: float = 0.0

@app.post("/predict/batch")
async def predict_batch(data: List[MatchInput]):
    # We use a list of 0.0 as a placeholder for now
    results = [0.0 for _ in data] 
    return {"predictions": results}
    
