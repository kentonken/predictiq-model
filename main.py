import os
import pandas as pd
from fastapi import FastAPI
from firecrawl import FirecrawlApp
from catboost import CatBoostClassifier
import requests

app = FastAPI()
firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
model = CatBoostClassifier().load_model("catboost_model.cbm")

# --- STEP 1: Get Basic Info from TheSportsDB ---
def get_fixture_info(match_id):
    # Free Tier uses '1' as key, or use your Premium Key
    url = f"https://www.thesportsdb.com/api/v1/json/3/lookupevent.php?id={match_id}"
    data = requests.get(url).json()
    return data['events'][0]

# --- STEP 2: Scrape Deep Stats with Firecrawl ---
def scrape_match_stats(url):
    # Firecrawl turns a website into clean JSON for your model
    scraped_data = firecrawl.scrape_url(url, params={
        'extractor_options': {
            'extraction_schema': {
                'type': 'object',
                'properties': {
                    'home_possession': {'type': 'number'},
                    'away_possession': {'type': 'number'},
                    'home_shots': {'type': 'number'},
                    'away_shots': {'type': 'number'}
                }
            }
        }
    })
    return scraped_data['data']

@app.post("/predict/smart")
async def smart_predict(match_id: str, scrape_url: str):
    # 1. Get metadata
    base_info = get_fixture_info(match_id)
    
    # 2. Get live stats
    stats = scrape_match_stats(scrape_url)
    
    # 3. Combine and Predict
    # (Ensure the dictionary keys match your MatchInput class from before)
    final_features = pd.DataFrame([{
        "home_league_position": int(base_info.get("intHomeLeaguePos", 0)),
        "home_pts": int(base_info.get("intHomePoints", 0)),
        "home_possession": stats["home_possession"],
        # ... add all other features here
    }])
    
    prob = model.predict_proba(final_features)[0][1]
    return {"home_win_probability": float(prob)}
    
