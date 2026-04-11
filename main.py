import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import logging

# Corrected import to match predictor.py and fix the 'ImportError'
from predictor import predict as run_predict, get_model

app = FastAPI(title="PredictIQ Pro API", version="5.1.0")
_training_status = {"status": "idle", "message": "Ready"}

class PredictionRequest(BaseModel):
    # These match your Lovable 'Railway CatBoost' test payload
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

class TrainRequest(BaseModel):
    date_from: str = "2026-04-08" # Use short ranges to avoid Railway RAM crashes
    date_to: str = "2026-04-11"
    secret: str

@app.post("/predict")
async def predict_endpoint(req: PredictionRequest):
    try:
        # Passes the scanner data to your v5 Ensemble logic
        return run_predict(req.dict())
    except Exception as e:
        # Returns 500 if model file is missing or training failed
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if req.secret != os.getenv("TRAIN_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid Secret")
    
    global _training_status
    _training_status = {"status": "running", "message": "Training V5 Ensemble..."}
    # background_tasks.add_task(your_training_logic)
    return {"message": "Training started. Check /train/status."}

@app.get("/train/status")
async def get_status():
    return _training_status

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model_loaded": get_model() is not None, 
        "region": "Uganda"
    }
    
