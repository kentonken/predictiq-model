from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI
import pandas as pd
from catboost import CatBoostClassifier
import os

app = FastAPI()

# 1. Load the model
MODEL_PATH = "catboost_model.cbm"
model = CatBoostClassifier()

if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print(f"CRITICAL ERROR: {MODEL_PATH} not found in repository.")

# 2. Match the data structure your model expects
class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_possession: float = 50.0
    away_possession: float = 50.0
    home_shots: float = 0.0
    away_shots: float = 0.0
    avg_goals_last_5: float = 0.0

@app.post("/predict/batch")
async def predict_batch(data: List[MatchInput]):
    try:
        # Convert incoming JSON to a list of dictionaries
        input_list = [item.dict() for item in data]
        df = pd.DataFrame(input_list)
        
        # IMPORTANT: Remove columns the model wasn't trained on (like names)
        # Your model likely only wants the numbers:
        features = df.drop(columns=['home_team', 'away_team'])
        
        # Get the probability of a Home Win (usually index 1)
        probs = model.predict_proba(features)
        
        # Convert to a standard list of floats so JSON can read it
        predictions = [float(p[1]) for p in probs]
        
        return {"predictions": predictions}
    
    except Exception as e:
        return {"error": str(e)}
        
