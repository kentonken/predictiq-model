from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from predictor import predict as run_predict, get_model

app = FastAPI(title="PredictIQ Pro API", version="5.0.0")

class PredictionRequest(BaseModel):
    # Field names must match your Lovable JSON exactly to avoid 422 errors
    home_team: str
    away_team: str
    league: str
    country: str
    home_attack_strength: float
    away_attack_strength: float
    home_defense_weakness: float
    away_defense_weakness: float
    home_form_score: float
    away_form_score: float
    home_consistency: float
    away_consistency: float
    poisson_lambda_home: float
    poisson_lambda_away: float
    poisson_over25_prob: float
    poisson_btts_p: float
    league_id: Optional[int] = 0

@app.post("/predict")
async def predict_endpoint(req: PredictionRequest):
    try:
        return run_predict(req.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "v5_active": get_model() is not None,
        "region": "Uganda"
    }
    
