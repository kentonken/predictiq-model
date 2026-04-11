import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import logging

# This fixes the 'cannot import name get_model' crash from your logs
from predictor import predict as run_predict, get_model

app = FastAPI(title="PredictIQ Pro API", version="5.1.0")
_training_status = {"status": "idle", "message": "Ready"}

class TrainRequest(BaseModel):
    date_from: str = "2025-01-01"
    date_to: str = "2026-04-10"
    secret: str

class PredictionRequest(BaseModel):
    # Removing '= 0' or '= 1500' forces your scraper to send REAL data
    league_id: int
    home_elo: float
    away_elo: float
    home_form_pts_5: float
    away_form_pts_5: float
    home_xg_for_5: float
    away_xg_for_5: float

# --- Endpoints ---

@app.post("/train")
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if req.secret != os.getenv("TRAIN_SECRET"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # This will show up in your Swagger UI once the app starts successfully
    background_tasks.add_task(print, "Training logic would run here...") 
    return {"message": "Training started. Check /train/status for updates."}

@app.get("/train/status")
async def get_status():
    return _training_status

@app.post("/predict")
async def predict_endpoint(req: PredictionRequest):
    try:
        # Uses the ensemble stacking logic from predictor.py
        return run_predict(req.dict())
    except Exception as e:
        # This will return the "Model not found" message if training isn't done
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model_active": get_model() is not None}
    
