import numpy as np

def predict_match(model, features, fid):
    if model is None:
        return {"error": "Model not loaded", "fixture_id": fid}

    try:
        # Get the class prediction (0, 1, or 2)
        prediction = model.predict(features)
        # Get the probabilities for each class
        probs = model.predict_proba(features)
        
        # Ensure we are getting a scalar index
        pred_idx = int(prediction[0]) if hasattr(prediction, "__len__") else int(prediction)
        
        # Map to human-readable outcomes
        outcomes = ["Home Win", "Draw", "Away Win"]
        prediction_text = outcomes[pred_idx]
        
        # Confidence is the highest probability in the array
        confidence = float(np.max(probs))

        return {
            "fixture_id": fid,
            "prediction": prediction_text,
            "confidence": confidence,
            "status": "success",
            "tip": f"AI Pick: {prediction_text}"
        }
    except Exception as e:
        return {"error": str(e), "fixture_id": fid}
        
