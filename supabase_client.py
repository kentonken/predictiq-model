import os
from supabase import create_client

def save_prediction(prediction: dict):
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    db = create_client(url, key)

    m_date = prediction.get("match_date")
    m_time = prediction.get("match_time")

    # Combine into a standard ISO timestamp for Supabase TIMESTAMPTZ columns
    # This specifically fixes the "TBD" issue in the UI
    full_timestamp = f"{m_date}T{m_time}:00Z" if m_date and m_time else None

    payload = {
        "fixture_id": str(prediction.get("fixture_id")),
        "home_team": prediction.get("home_team"),
        "away_team": prediction.get("away_team"),
        "home_logo": prediction.get("home_logo"),
        "away_logo": prediction.get("away_logo"),
        "match_time": full_timestamp,  # Fixed timestamp
        "prediction": prediction.get("prediction"),
        "confidence": prediction.get("confidence"),
        "league": prediction.get("league"),
        "tip": prediction.get("tip")
    }

    # Upsert ensures we update existing matches instead of creating duplicates
    db.table("predictions").upsert(payload, on_conflict="fixture_id").execute()
    
