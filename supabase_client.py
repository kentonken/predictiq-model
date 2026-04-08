import os
from supabase import create_client, Client

def save_prediction(prediction: dict):
    db = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    
    m_date = prediction.get("match_date")
    m_time = prediction.get("match_time")
    
    # Final fix for "TBD": Creates ISO-8601 string for Lovable
    full_timestamp = f"{m_date}T{m_time}:00Z" if m_date and m_time else None

    db.table("predictions").upsert({
        "fixture_id": str(prediction.get("fixture_id")),
        "home_team": prediction.get("home_team"),
        "away_team": prediction.get("away_team"),
        "home_logo": prediction.get("home_logo"),
        "away_logo": prediction.get("away_logo"),
        "match_time": full_timestamp, # Populates UI clock
        "odds_1x2": prediction.get("odds_1x2"),
        "prediction": prediction.get("prediction"),
        "confidence": prediction.get("confidence"),
        "league": prediction.get("league"),
        "tip": prediction.get("tip")
    }, on_conflict="fixture_id").execute()
    
