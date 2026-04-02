import os
import requests
from typing import Optional

# Configuration for Bzzoiro (BSD)
# Note: Ensure you set 'BZZOIRO_API_KEY' in your environment variables.
API_KEY = os.getenv("BZZOIRO_API_KEY")
BASE_URL = "https://sports.bzzoiro.com/api/v1"

# Bzzoiro typically uses a standard API Key header
HEADERS = {
    "X-Api-Key": API_KEY,
    "Accept": "application/json"
}

def _get(endpoint: str, params: dict = None) -> dict:
    if not API_KEY:
        raise Exception("BZZOIRO_API_KEY environment variable is not set")
    
    url = f"{BASE_URL}/{endpoint}"
    res = requests.get(url, headers=HEADERS, params=params)
    res.raise_for_status()
    return res.json()

def get_fixture(fixture_id: int) -> Optional[dict]:
    # Bzzoiro football endpoints follow the standard 'response' array structure
    data = _get("fixtures", {"id": fixture_id})
    fixtures = data.get("response", [])
    return fixtures[0] if fixtures else None

def get_team_last_home_fixtures(team_id: int, count: int = 5) -> list:
    # Fetching recent matches to filter for home games
    data = _get("fixtures", {"team": team_id, "last": 20})
    response = data.get("response", [])
    home = [f for f in response if f["teams"]["home"]["id"] == team_id]
    return home[:count]

def get_team_last_away_fixtures(team_id: int, count: int = 5) -> list:
    # Fetching recent matches to filter for away games
    data = _get("fixtures", {"team": team_id, "last": 20})
    response = data.get("response", [])
    away = [f for f in response if f["teams"]["away"]["id"] == team_id]
    return away[:count]

def get_head_to_head(home_team_id: int, away_team_id: int, count: int = 5) -> list:
    params = {
        "h2h": f"{home_team_id}-{away_team_id}",
        "last": count
    }
    data = _get("fixtures/headtohead", params)
    return data.get("response", [])

def get_standings(league_id: int, season: int) -> list:
    data = _get("standings", {"league": league_id, "season": season})
    try:
        # Standard nesting: response -> league -> standings -> table
        return data["response"][0]["league"]["standings"][0]
    except (IndexError, KeyError):
        return []

def get_injuries(fixture_id: int) -> list:
    data = _get("injuries", {"fixture": fixture_id})
    return data.get("response", [])
    
