import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

_client: Client = None

def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise Exception("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client

def save_prediction(prediction: dict):
    db = get_client()
    
    # Extract date and time from the prediction dictionary
    m_date = prediction.get("match_date")
    m_time = prediction.get("match_time")
    
    # Format the timestamp for the 'timestamp with time zone' column
    # This prevents the 'NULL' value that causes "TBD" in Lovable
    full_timestamp = f"{m_date}T{m_time}:00Z" if m_date and m_time else None

    # Note: Using 'predictions' as the table name based on your Supabase screenshots
    db.table("predictions").upsert({
        "fixture_id": str(prediction.get("fixture_id")),
        "home_team": prediction.get("home_team"),
        "away_team": prediction.get("away_team"),
        "prediction": prediction.get("prediction"),
        "confidence": prediction.get("confidence"),
        "home_win_prob": prediction.get("home_win_prob"),
        "draw_prob": prediction.get("draw_prob"),
        "away_win_prob": prediction.get("away_win_prob"),
        "tip": prediction.get("tip"),
        "match_date": m_date,
        "match_time": full_timestamp,
        "league": prediction.get("league"),
        "country": prediction.get("country")
    }, on_conflict="fixture_id").execute()

def get_prediction(fixture_id: int) -> dict | None:
    db = get_client()
    try:
        res = db.table("predictions") \
            .select("*") \
            .eq("fixture_id", str(fixture_id)) \
            .single() \
            .execute()
        return res.data if res.data else None
    except Exception:
        return None
        
