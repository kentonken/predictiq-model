from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predictor import load_model, predict_match
from features import compute_features
from supabase_client import save_prediction, get_prediction

app = FastAPI(title="PredictIQ Model API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

class PredictRequest(BaseModel):
    fixture_id: int
    use_cache: bool = True

@app.get("/")
def root():
    return {"status": "PredictIQ Model API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    if req.use_cache:
        cached = get_prediction(req.fixture_id)
        if cached:
            cached["cached"] = True
            return cached
            
    features = compute_features(req.fixture_id)
    if not features:
        raise HTTPException(status_code=404, detail="Fixture not found")
        
    result = predict_match(model, features, req.fixture_id)
    
    try:
        save_prediction(result)
    except Exception as e:
        print(f"Could not save to Supabase: {e}")
        
    result["cached"] = False
    return result

@app.post("/predict/batch")
def predict_batch(fixture_ids: list[int]):
    if len(fixture_ids) > 20:
        raise HTTPException(status_code=400, detail="Max 20 fixtures per batch")
        
    results = []
    errors = []
    
    for fid in fixture_ids:
        try:
            cached = get_prediction(fid)
            if cached:
                cached["cached"] = True
                results.append(cached)
                continue
                
            features = compute_features(fid)
            if not features:
                errors.append({"fixture_id": fid, "error": "Not found"})
                continue
                
            result = predict_match(model, features, fid)
            save_prediction(result)
            result["cached"] = False
            results.append(result)
            
        except Exception as e:
            errors.append({"fixture_id": fid, "error": str(e)})
            
    return {"predictions": results, "errors": errors}
        
