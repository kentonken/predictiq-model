import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

_client: Client = None

def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise Exception("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in Environment Variables")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client

def save_prediction(prediction: dict):
    db = get_client()
    
    # 1. Format Kickoff Time to fix "TBD" issue
    m_date = prediction.get("match_date")
    m_time = prediction.get("match_time")
    full_timestamp = f"{m_date}T{m_time}:00Z" if m_date and m_time else None

    # 2. Push all data including Logos and Odds
    db.table("predictions").upsert({
        "fixture_id": str(prediction.get("fixture_id")),
        "home_team": prediction.get("home_team"),
        "away_team": prediction.get("away_team"),
        
        # LOGOS & CODES
        "home_logo": prediction.get("home_logo"),
        "away_logo": prediction.get("away_logo"),
        "home_code": prediction.get("home_code"),
        "away_code": prediction.get("away_code"),

        # ODDS & BOOKMAKER
        "bookmaker": prediction.get("bookmaker", "Bzzoiro"),
        "odds_1x2": prediction.get("odds_1x2"),
        "odds_over25": prediction.get("odds_over25"),
        "odds_under25": prediction.get("odds_under25"),
        "odds_btts": prediction.get("odds_btts"),

        # MODEL PREDICTIONS
        "prediction": prediction.get("prediction"),
        "confidence": prediction.get("confidence"),
        "home_win_prob": prediction.get("home_win_prob"),
        "draw_prob": prediction.get("draw_prob"),
        "away_win_prob": prediction.get("away_win_prob"),
        "tip": prediction.get("tip"),
        "is_value_bet": prediction.get("is_value_bet", False),

        # KICKOFF & LEAGUE INFO
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
        
