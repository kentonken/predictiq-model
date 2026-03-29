import os
import requests
from typing import Optional

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

def _get(endpoint: str, params: dict) -> dict:
    if not API_KEY:
        raise Exception("API_FOOTBALL_KEY environment variable is not set")
    url = f"{BASE_URL}/{endpoint}"
    if not (url.startswith("https://") or url.startswith("http://")):
        raise ValueError("Invalid URL scheme")
    res = requests.get(url, headers=HEADERS, params=params, timeout=10)
    res.raise_for_status()
    return res.json()

def get_fixture(fixture_id: int) -> Optional[dict]:
    data = _get("fixtures", {"id": fixture_id})
    fixtures = data.get("response", [])
    return fixtures[0] if fixtures else None

def get_team_last_home_fixtures(team_id: int, count: int = 5) -> list:
    data = _get("fixtures", {"team": team_id, "last": 20})
    home = [f for f in data.get("response", []) if f["teams"]["home"]["id"] == team_id]
    return home[:count]

def get_team_last_away_fixtures(team_id: int, count: int = 5) -> list:
    data = _get("fixtures", {"team": team_id, "last": 20})
    away = [f for f in data.get("response", []) if f["teams"]["away"]["id"] == team_id]
    return away[:count]

def get_head_to_head(home_team_id: int, away_team_id: int, count: int = 5) -> list:
    data = _get("fixtures/headtohead", {
        "h2h": f"{home_team_id}-{away_team_id}",
        "last": count
    })
    return data.get("response", [])

def get_standings(league_id: int, season: int) -> list:
    data = _get("standings", {"league": league_id, "season": season})
    try:
        return data["response"][0]["league"]["standings"][0]
    except (IndexError, KeyError):
        return []

def get_injuries(fixture_id: int) -> list:
    data = _get("injuries", {"fixture": fixture_id})
    return data.get("response", [])
