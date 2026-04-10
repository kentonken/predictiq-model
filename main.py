from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import os
import pandas as pd
import traceback

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PredictIQ Pro ML API", version="5.0.0")

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared training status to track progress via /train/status
_training_status = {"status": "idle", "message": "API is online. Ready for initial training."}

@app.on_event("startup")
async def startup():
    """
    Attempts to load the ensemble model from Supabase Storage on container startup.
    """
    try:
        from predictor import get_model
        model = get_model()
        if model:
            logger.info("✅ PredictIQ Pro: Model loaded successfully from Supabase.")
            global _training_status
            _training_status = {"status": "complete", "message": "Model loaded and active."}
        else:
            logger.warning("⚠️ PredictIQ Pro: No model found in Storage. Use /train to generate one.")
    except Exception as e:
        logger.error(f"❌ Startup Error: {e}")

# --- Feature Schema (163 Features) ---
class PredictionRequest(BaseModel):
    # Core Match Info
    league_id: int = 0
    league_tier: int = 1
    match_week: int = 20
    season_progress: float = 0.5
    is_neutral_venue: int = 0
    is_derby: int = 0
    # Elo & Standings
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    home_league_pos: int = 8
    away_league_pos: int = 12
    # Form & Win Rates
    home_win_rate_5: float = 0.5
    away_win_rate_5: float = 0.4
    home_draw_rate_5: float = 0.2
    away_draw_rate_5: float = 0.2
    home_loss_rate_5: float = 0.3
    away_loss_rate_5: float = 0.4
    home_form_pts_5: float = 9.0
    away_form_pts_5: float = 6.0
    home_win_rate_10: float = 0.5
    away_win_rate_10: float = 0.4
    home_prev_match_result: int = 1
    away_prev_match_result: int = 1
    # Goals & xG
    home_goals_scored_5: float = 1.5
    away_goals_scored_5: float = 1.1
    home_goals_conceded_5: float = 1.0
    away_goals_conceded_5: float = 1.4
    home_xg_for_5: float = 1.6
    away_xg_for_5: float = 1.2
    home_xg_against_5: float = 1.0
    away_xg_against_5: float = 1.5
    home_btts_rate: float = 0.5
    away_btts_rate: float = 0.5
    # H2H Statistics
    h2h_home_wins: int = 3
    h2h_draws: int = 2
    h2h_away_wins: int = 1
    h2h_avg_goals: float = 2.5
    h2h_btts_rate: float = 0.5
    # Defensive/Clean Sheet
    home_over25_rate: float = 0.5
    away_over25_rate: float = 0.5
    home_clean_sheet_5: float = 0.3
    away_clean_sheet_5: float = 0.3
    home_scored_both_5: float = 0.6
    away_scored_both_5: float = 0.6
    # Advanced Poisson & Shots
    poisson_home_goals: float = 1.4
    poisson_away_goals: float = 1.1
    home_shots_on_target_5: float = 4.5
    away_shots_on_target_5: float = 3.5
    home_total_shots_5: float = 12.0
    away_total_shots_5: float = 10.0
    home_shot_conversion_5: float = 0.12
    away_shot_conversion_5: float = 0.10
    home_xg_per_shot_5: float = 0.11
    away_xg_per_shot_5: float = 0.10
    home_big_chances_5: float = 2.5
    away_big_chances_5: float = 2.0
    home_shots_box_pct_5: float = 0.55
    away_shots_box_pct_5: float = 0.50
    home_shots_first_half_5: float = 5.5
    away_shots_first_half_5: float = 4.5
    home_shots_second_half_5: float = 7.0
    away_shots_second_half_5: float = 5.5
    home_post_hits_5: float = 0.8
    away_post_hits_5: float = 0.7
    home_penalty_rate: float = 0.08
    away_penalty_rate: float = 0.07
    home_avg_shot_x: float = 85.0
    away_avg_shot_x: float = 83.0
    home_avg_shot_xg: float = 0.12
    away_avg_shot_xg: float = 0.10
    home_xgot_ratio: float = 1.1
    away_xgot_ratio: float = 1.0
    home_header_goals_rate: float = 0.20
    away_header_goals_rate: float = 0.18
    home_set_piece_goals_rate: float = 0.25
    away_set_piece_goals_rate: float = 0.22
    # Momentum & Pressure
    home_momentum_avg_5: float = 5.0
    away_momentum_avg_5: float = -5.0
    home_momentum_peak_5: float = 25.0
    away_momentum_peak_5: float = 20.0
    home_momentum_variance_5: float = 12.0
    away_momentum_variance_5: float = 10.0
    home_pressure_index_5: float = 0.55
    away_pressure_index_5: float = 0.45
    home_late_pressure_5: float = 0.6
    away_late_pressure_5: float = 0.5
    home_comeback_rate: float = 0.2
    away_comeback_rate: float = 0.15
    home_goals_min_75_90_5: float = 0.4
    away_goals_min_75_90_5: float = 0.35
    # Tactical Positional Data
    home_avg_def_line_x: float = 35.0
    away_avg_def_line_x: float = 35.0
    home_avg_att_line_x: float = 75.0
    away_avg_att_line_x: float = 75.0
    home_team_width: float = 55.0
    away_team_width: float = 55.0
    home_team_compactness: float = 40.0
    away_team_compactness: float = 40.0
    home_pressing_intensity: float = 0.5
    away_pressing_intensity: float = 0.5
    # Possession & Passing
    home_possession_avg_5: float = 52.0
    away_possession_avg_5: float = 48.0
    home_passes_per_game_5: float = 450.0
    away_passes_per_game_5: float = 420.0
    home_crossing_rate_5: float = 0.25
    away_crossing_rate_5: float = 0.22
    home_tackles_won_5: float = 18.0
    away_tackles_won_5: float = 16.0
    home_interceptions_5: float = 12.0
    away_interceptions_5: float = 11.0
    home_aerials_won_5: float = 22.0
    away_aerials_won_5: float = 20.0
    # Discipline & Corners
    home_fouls_conceded_5: float = 11.0
    away_fouls_conceded_5: float = 12.0
    home_yellow_cards_5: float = 1.5
    away_yellow_cards_5: float = 1.8
    home_red_cards_5: float = 0.1
    away_red_cards_5: float = 0.1
    home_corners_for_5: float = 5.5
    away_corners_for_5: float = 4.5
    # Squad & Availability
    home_squad_depth_score: float = 0.7
    away_squad_depth_score: float = 0.7
    home_injuries_count: int = 2
    away_injuries_count: int = 2
    home_season_goals_scored: int = 28
    away_season_goals_scored: int = 22
    days_since_last_home: int = 7
    days_since_last_away: int = 7
    # Bzzoiro Probability Indicators
    bzz_prob_home_win: float = 45.0
    bzz_prob_draw: float = 25.0
    bzz_prob_away_win: float = 30.0
    bzz_expected_home_goals: float = 1.4
    bzz_expected_away_goals: float = 1.1
    bzz_prob_over_15: float = 75.0
    bzz_prob_over_25: float = 55.0
    bzz_prob_over_35: float = 30.0
    bzz_prob_btts: float = 50.0
    bzz_confidence: float = 0.65
    bzz_home_win_recommend: int = 0
    bzz_over_15_recommend: int = 1
    bzz_over_25_recommend: int = 1
    bzz_over_35_recommend: int = 0
    bzz_btts_recommend: int = 1
    bzz_winner_recommend: int = 1
    bzz_model_version_enc: int = 5

# --- Background Training Job ---
def _run_training_job(date_from: str, date_to: str):
    global _training_status
    _training_status = {"status": "running", "message": "Downloading match data from Bzzoiro..."}
    
    try:
        from api_bzzoiro import get_finished_predictions_for_training
        from features import engineer_features
        from train import EnsembleStacker
        from model_store import upload_model
        import predictor

        # 1. Fetch historical data
        rows = get_finished_predictions_for_training(date_from, date_to)
        if len(rows) < 5:
            _training_status = {"status": "error", "message": "Not enough match data found."}
            return

        df = pd.DataFrame(rows)
        _training_status["message"] = f"Training ensemble model on {len(df)} matches..."

        # 2. Train and fit
        X = engineer_features(df)
        y = df["result"]
        model = EnsembleStacker()
        model.fit(X, y)

        # 3. Save to Supabase and update local reference
        _training_status["message"] = "Uploading ensemble_v5.pkl to Supabase..."
        if upload_model(model):
            predictor._model = model 
            _training_status = {"status": "complete", "message": "New model deployed and active."}
        else:
            _training_status = {"status": "error", "message": "Supabase upload failed."}

    except Exception as e:
        logger.error(traceback.format_exc())
        _training_status = {"status": "error", "message": f"Training failed: {str(e)}"}

# --- API Endpoints ---

class TrainRequest(BaseModel):
    date_from: str = "2025-01-01"
    date_to: str = "2026-04-10"
    secret: str

@app.post("/train")
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Triggers the ML training pipeline. Requires TRAIN_SECRET from env vars.
    """
    if req.secret != os.getenv("TRAIN_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid secret key.")
    
    if _training_status["status"] == "running":
        return {"message": "A training job is currently in progress."}

    background_tasks.add_task(_run_training_job, req.date_from, req.date_to)
    return {"message": "Training started in the background. Monitor status via /train/status."}

@app.get("/train/status")
async def get_train_status():
    return _training_status

@app.post("/predict")
async def predict_match(req: PredictionRequest):
    """
    Returns AI outcomes based on the 163 provided features.
    """
    try:
        from predictor import predict as run_predict
        return run_predict(req.dict())
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "version": "5.0.0",
        "model_loaded": _training_status["status"] == "complete"
    }
    
