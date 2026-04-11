import os
import logging
import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import your predictor logic
from predictor import predict as run_predict, get_model

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PredictIQ Pro API", version="5.1.0")

# CORS for Lovable/Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_training_status = {"status": "idle", "message": "API is online."}

@app.on_event("startup")
async def startup():
    try:
        model = get_model()
        if model:
            logger.info("✅ Ensemble Model loaded successfully.")
            global _training_status
            _training_status["status"] = "complete"
    except Exception as e:
        logger.error(f"❌ Startup Error: {e}")

# --- STRICT SCHEMA (No Defaults) ---
# We remove "= 0" or "= 1.4" to force the API to use REAL data.
class PredictionRequest(BaseModel):
    league_id: int
    league_tier: int
    match_week: int
    season_progress: float
    is_neutral_venue: int
    is_derby: int
    
    # Core Team Strength
    home_elo: float
    away_elo: float
    home_league_pos: int
    away_league_pos: int
    
    # Form & Historicals (Critical for CatBoost)
    home_form_pts_5: float
    away_form_pts_5: float
    home_win_rate_10: float
    away_win_rate_10: float
    
    # Poisson Distribution Inputs
    # We keep these as Optional, but predictor.py must handle them if they are None.
    bzz_expected_home_goals: Optional[float] = None
    bzz_expected_away_goals: Optional[float] = None
    bzz_prob_home_win: Optional[float] = None
    bzz_prob_away_win: Optional[float] = None

# --- ENDPOINTS ---

@app.post("/predict")
async def predict(req: PredictionRequest):
    """
    Receives real match stats and passes them to the CatBoost 
    inference engine in predictor.py.
    """
    try:
        # Pass the validated dictionary to the predictor
        input_data = req.dict()
        results = run_predict(input_data)
        return results

    except Exception as e:
        logger.error(f"Inference Failed: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Predictor Error: {str(e)}"
        )

@app.get("/health")
async def health():
    return {"status": "ok", "model_active": get_model() is not None}

# --- Background Training ---
class TrainRequest(BaseModel):
    date_from: str = "2025-01-01"
    date_to: str = "2026-04-10"
    secret: str

@app.post("/train")
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if req.secret != os.getenv("TRAIN_SECRET"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Training logic remains the same...
    return {"message": "Training started."}
    
