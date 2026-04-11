import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from predictor import predict as run_predict, get_model

app = FastAPI(title="PredictIQ Pro API", version="5.1.0")

# --- Updated Schema to Match Your Screenshot ---
class PredictionRequest(BaseModel):
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
    # Optional field in case it's missing from the scan
    league_id: Optional[int] = 0 

class TrainRequest(BaseModel):
    date_from: str = "2026-04-08"
    date_to: str = "2026-04-11"
    secret: str

# --- Endpoints ---

@app.post("/predict")
async def predict_endpoint(req: PredictionRequest):
    try:
        # Passes the actual scanner data to the ensemble
        return run_predict(req.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if req.secret != os.getenv("TRAIN_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid Secret")
    # background_tasks.add_task(your_training_function)
    return {"message": "Micro-training started (3-day window)."}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": get_model() is not None,
        "region": "Uganda"
    }
    
