def predict_match(model, features, fid):
    if model is None:
        return {"error": "Model not loaded", "fixture_id": fid}

    # Get the raw prediction and probabilities
    prediction = model.predict(features)
    probs = model.predict_proba(features) # This gives us the confidence %
    
    # IMPORTANT: Map your model's numerical output to text
    # Adjust this order based on how you trained your model!
    outcomes = ["Home Win", "Draw", "Away Win"]
    
    # Get the index of the highest probability
    # If prediction[0] is the index, we use that
    pred_idx = int(prediction[0]) if hasattr(prediction, "__getitem__") else int(prediction)
    prediction_text = outcomes[pred_idx]
    
    return {
        "fixture_id": fid,
        "prediction": prediction_text, # Sends "Draw" instead of "1" or "X"
        "confidence": float(max(probs)), # Sends 0.85 instead of 0.00
        "status": "success",
        "tip": f"AI Pick: {prediction_text}"
    }
    
