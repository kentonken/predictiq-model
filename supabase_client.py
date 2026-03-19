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
    db.table("ai_predictions").upsert({
        "fixture_id": str(prediction["fixture_id"]),
        "home_team": prediction["home_team"],
        "away_team": prediction["away_team"],
        "prediction": prediction["prediction"],
        "confidence": prediction["confidence"],
        "home_win_prob": prediction["home_win_prob"],
        "draw_prob": prediction["draw_prob"],
        "away_win_prob": prediction["away_win_prob"],
        "tip": prediction["tip"],
    }, on_conflict="fixture_id").execute()

def get_prediction(fixture_id: int) -> dict | None:
    db = get_client()
    try:
        res = db.table("ai_predictions") \
            .select("*") \
            .eq("fixture_id", str(fixture_id)) \
            .single() \
            .execute()
        return res.data if res.data else None
    except:
        return None
