import os
from catboost import CatBoostClassifier

def load_model():
    # This must match the filename you uploaded to GitHub!
    model_path = os.path.join(os.getcwd(), "catboost_model.cbm")
    
    if os.path.exists(model_path):
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model
    else:
        print(f"ERROR: {model_path} not found in repository")
        return None

def predict_match(model, features, fid):
    if model is None:
        return {"error": "Model not loaded", "fixture_id": fid}
    
    # This assumes 'features' is a list or array your model expects
    prediction = model.predict(features)
    
    # CatBoost returns an array, we take the first value
    return {
        "fixture_id": fid,
        "prediction": str(prediction[0]) if hasattr(prediction, "__getitem__") else str(prediction),
        "status": "success"
    }
    
