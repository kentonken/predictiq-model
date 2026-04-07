import os
import requests
from typing import Optional

# Configuration for Bzzoiro (BSD)
API_KEY = os.getenv("BZZOIRO_API_KEY")
BASE_URL = "https://sports.bzzoiro.com/api/v1"

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

def get_fixture(league_id: int, season: int = 2025) -> list:
    """
    Fetches upcoming fixtures with odds and team logos included.
    """
    params = {
        "league": league_id,
        "season": season,
        # 'next' helps get upcoming matches for your daily predictions
        "next": 15,
        # Requesting odds specifically for the UI
        "odds": "bet365" 
    }
    data = _get("fixtures", params=params)
    return data.get("response", [])

def get_team_last_home_fixtures(team_id: int, count: int = 5) -> list:
    params = {"team": team_id, "last": 20}
    data = _get("fixtures", params=params)
    response = data.get("response", [])
    home = [f for f in response if f["teams"]["home"]["id"] == team_id]
    return home[:count]

def get_team_last_away_fixtures(team_id: int, count: int = 5) -> list:
    params = {"team": team_id, "last": 20}
    data = _get("fixtures", params=params)
    response = data.get("response", [])
    away = [f for f in response if f["teams"]["away"]["id"] == team_id]
    return away[:count]

def get_head_to_head(home_team_id: int, away_team_id: int, count: int = 5) -> list:
    params = {
        "h2h": f"{home_team_id}-{away_team_id}",
        "last": count
    }
    data = _get("fixtures/headtohead", params=params)
    return data.get("response", [])

def get_standings(league_id: int, season: int = 2025) -> list:
    params = {"league": league_id, "season": season}
    try:
        data = _get("standings", params=params)
        return data["response"][0]["league"]["standings"][0]
    except (IndexError, KeyError):
        return []

def get_injuries(fixture_id: int) -> list:
    data = _get("injuries", params={"fixture": fixture_id})
    return data.get("response", [])
    
