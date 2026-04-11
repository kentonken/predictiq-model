import os
import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from predictor import predict as run_predict, get_model

app = FastAPI(title="PredictIQ Pro API")

# --- Prediction Schema (No Defaults) ---
class PredictionRequest(BaseModel):
    # Match Meta
    league_id: int
    home_elo: float
    away_elo: float
    
    # Required Stats (Removing '= 1500.0' etc forces the frontend to send data)
    home_form_pts_5: float
    away_form_pts_5: float
    home_xg_for_5: float
    away_xg_for_5: float
    
    # Optional Bzzoiro Data
    bzz_expected_home_goals: Optional[float] = None
    bzz_expected_away_goals: Optional[float] = None

@app.post("/predict")
async def predict(req: PredictionRequest):
    try:
        # 1. Convert request to dict
        input_dict = req.dict()
        
        # 2. Run the predictor logic
        # This will now use the REAL values from the dict, not 1500.0
        results = run_predict(input_dict)
        
        return results
    except Exception as e:
        logger.error(f"Inference Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
        
