from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI  # <--- ADD THIS

app = FastAPI() # <--- ADD THIS

# This makes the input flexible so it doesn't crash on missing data
class MatchInput(BaseModel):
    home_team: str
    away_team: str
    # ... (the rest of your class code)

@app.post("/predict/batch")
async def predict_batch(data: List[MatchInput]):
    # Your logic goes here
    return {"predictions": [...]}
    
