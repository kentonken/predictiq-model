from pydantic import BaseModel
from typing import List, Optional

# This makes the input flexible so it doesn't crash on missing data
class MatchInput(BaseModel):
    home_team: str
    away_team: str
    # Using float = 0.0 prevents the 422 error if data is decimal or missing
    home_possession: float = 50.0
    away_possession: float = 50.0
    home_shots: float = 0.0
    away_shots: float = 0.0
    avg_goals_last_5: float = 0.0

@app.post("/predict/batch")
async def predict_batch(data: List[MatchInput]):
    # Your logic to run catboost_model.cbm goes here
    # Return probabilities to Lovable
    return {"predictions": [...]}
    
