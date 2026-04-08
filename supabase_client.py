import os
from supabase import create_client

def save_prediction(prediction: dict):
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    db = create_client(url, key)
    
    m_date = prediction.get("match_date")
    m_time = prediction.get("match_time")
    
    # Standard ISO format that triggers the Lovable UI to show a real clock
    full_timestamp = f"{m_date}T{m_time}:00Z" if m_date and m_time else None

    db.table("predictions").upsert({
        "fixture_id": str(prediction.get("fixture_id")),
        "home_team": prediction.get("home_team"),
        "away_team": prediction.get("away_team"),
        "home_logo": prediction.get("home_logo"),
        "away_logo": prediction.get("away_logo"),
        "match_time": full_timestamp, # Fixes TBD
        "odds_1x2": prediction.get("odds_1x2"), # Fixes 0.00
        "prediction": prediction.get("prediction"),
        "confidence": prediction.get("confidence"), # Decimal format (e.g. 0.85)
        "league": prediction.get("league"),
        "tip": prediction.get("tip")
    }, on_conflict="fixture_id").execute()
    
