import os
import requests

API_KEY = os.getenv("BZZOIRO_API_KEY")
BASE_URL = "https://sports.bzzoiro.com/api/v1"
HEADERS = {"X-Api-Key": API_KEY, "Accept": "application/json"}

def _get(endpoint: str, params: dict = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    res = requests.get(url, headers=HEADERS, params=params)
    res.raise_for_status()
    return res.json()

def get_fixture(league_id: int, season: int = 2025) -> list:
    params = {
        "league": league_id,
        "season": season,
        "next": 15,
        "odds": "bet365"  # Fetches the odds you need for the UI
    }
    data = _get("fixtures", params=params)
    return data.get("response", [])
    
