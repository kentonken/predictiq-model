import os
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

# CRITICAL: This fixes your 'ImportError' by matching predictor.py exactly
from predictor import predict as run_predict, get_model

app = FastAPI(title="PredictIQ Pro API", version="0.1.0")

# Internal status tracking
_training_status = {"status": "idle", "message": "Ready"}

class TrainRequest(BaseModel):
    date_from: str = "2025-01-01"
    date_to: str = "2026-04-10"
    secret: str

class PredictionRequest(BaseModel):
    # Removing '= 0' forces the API to require REAL data from your scraper
    league_id: int
    home_elo: float
    away_elo: float
    home_form_pts_5: float
    away_form_pts_5: float
    home_xg_for_5: float
    away_xg_for_5: float

@app.post("/train")
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    # Check your Railway environment variables for TRAIN_SECRET
    if req.secret != os.getenv("TRAIN_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid Secret")
    
    global _training_status
    _training_status = {"status": "running", "message": "Fetching Bzzoiro data..."}
    
    # background_tasks.add_task(your_actual_training_function)
    return {"message": "Training started. Check /train/status for progress."}

@app.get("/train/status")
async def get_status():
    return _training_status

@app.post("/predict")
async def predict_endpoint(req: PredictionRequest):
    try:
        # Calls the dynamic logic in predictor.py
        return run_predict(req.dict())
    except Exception as e:
        # Returns the 'Model not found' error if it hasn't been trained yet
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model_loaded": get_model() is not None,
        "region": "Uganda"
    }
    
