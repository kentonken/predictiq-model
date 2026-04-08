import os
import requests
from supabase import create_client

# 1. Configuration (Set these in Railway Variables)
API_KEY = os.getenv("BZZOIRO_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Correct Auth for Bzzoiro
BASE_URL = "https://sports.bzzoiro.com/api"
HEADERS = {
    "Authorization": f"Token {API_KEY}",
    "Accept": "application/json"
}

# Initialize Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def sync_bzzoiro_predictions():
    """Fetches ML predictions and saves to ai_predictions table."""
    url = f"{BASE_URL}/predictions/"
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    predictions = res.json()

    for pred in predictions:
        # Map Bzzoiro fields to your Supabase ai_predictions schema
        data = {
            "fixture_id": str(pred['event_id']),
            "home_win_prob": pred['prob_home_win'],
            "draw_prob": pred['prob_draw'],
            "away_win_prob": pred['prob_away_win'],
            "confidence": pred['confidence'],
            "prediction": pred['most_likely_score'],
            "bzzoiro_ml_data": pred  # Store the full JSON for the heatmap/other features
        }
        supabase.table("ai_predictions").upsert(data, on_conflict="fixture_id").execute()
    return f"Synced {len(predictions)} predictions."

def sync_team_squad(team_id: int):
    """Fetches squad and profile images for a specific team."""
    url = f"{BASE_URL}/players/?team={team_id}"
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    players = res.json()

    for p in players:
        player_data = {
            "id": p['id'],
            "name": p['name'],
            "team_id": team_id,
            "rating": p.get('rating', 0),
            "photo_url": f"https://sports.bzzoiro.com/img/player/{p['id']}/"
        }
        supabase.table("players").upsert(player_data).execute()
    return f"Synced {len(players)} players for team {team_id}."

def get_player_stats(player_id: int, fixture_id: str):
    """Syncs 30+ metrics and heatmap coords for a player."""
    url = f"{BASE_URL}/player-stats/?player={player_id}"
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    all_stats = res.json()

    if all_stats:
        # Assuming index 0 is the most recent match
        latest = all_stats[0]
        perf_data = {
            "player_id": player_id,
            "fixture_id": fixture_id,
            "heatmap_data": latest.get('heatmap_coords'),
            "metrics": latest
        }
        supabase.table("player_match_stats").upsert(perf_data).execute()

if __name__ == "__main__":
    # Run the main sync
    print(sync_bzzoiro_predictions())
    
