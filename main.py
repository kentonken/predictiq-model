from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from predictor import predict as run_predict, get_model # This matches the fix above

app = FastAPI(title="PredictIQ Pro API")

class PredictionRequest(BaseModel):
    # Removing '= 1500.0' forces the API to use REAL data
    league_id: int
    home_elo: float
    away_elo: float
    home_form_pts_5: float
    away_form_pts_5: float
    home_xg_for_5: float
    away_xg_for_5: float
    bzz_expected_home_goals: Optional[float] = None
    bzz_expected_away_goals: Optional[float] = None

@app.post("/predict")
async def predict(req: PredictionRequest):
    try:
        # Pass real data to the ensemble
        return run_predict(req.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
