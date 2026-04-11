from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# CRITICAL: This line must match your predictor.py function names
from predictor import predict as run_predict, get_model 

app = FastAPI(title="PredictIQ Pro API")

class PredictionRequest(BaseModel):
    league_id: int
    home_elo: float
    away_elo: float
    # Add other required fields...

@app.post("/predict")
async def predict_endpoint(req: PredictionRequest):
    try:
        # This calls the 'run_predict' (renamed from 'predict') in predictor.py
        return run_predict(req.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
