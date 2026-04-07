import os
from api_bzzoiro import get_fixture
from predictor import predict_match, load_model
from supabase_client import save_prediction

def run_daily_predictions():
    # 1. Load the CatBoost model from the repository
    model = load_model()
    if not model:
        print("Error: Model could not be loaded. Check catboost_model.cbm.")
        return

    # 2. Targeted "Hidden Gem" Leagues (e.g., Poland, Kazakhstan, etc.)
    # Add your specific strategy league IDs here
    leagues = [348, 350, 281, 282] 
    
    for league_id in leagues:
        print(f"Fetching fixtures for league: {league_id}")
        matches = get_fixture(league_id) 
        
        if not matches:
            print(f"No matches found for league {league_id}")
            continue
            
        for match in matches:
            fixture = match.get('fixture', {})
            teams = match.get('teams', {})
            league_info = match.get('league', {})
            odds = match.get('odds', {}) 

            # 3. Handle Kickoff Time to fix "TBD" in Lovable UI
            raw_date = fixture.get('date') # Format: 2026-04-07T18:45:00+00:00
            m_date = raw_date.split('T')[0] if raw_date else None
            m_time = raw_date.split('T')[1][:5] if raw_date else None

            # 4. Prepare Features & Generate Prediction
            # Ensure this matches the feature order in your training logic
            try:
                # You'll need to pass the actual feature list your model expects here
                features = [] 
                prediction_results = predict_match(model, features, fixture.get('id'))
            except Exception as e:
                print(f"Prediction failed for fixture {fixture.get('id')}: {e}")
                continue

            # 5. Construct the Full Payload for Supabase
            final_data = {
                "fixture_id": fixture.get('id'),
                "match_date": m_date,
                "match_time": m_time,
                "home_team": teams.get('home', {}).get('name'),
                "away_team": teams.get('away', {}).get('name'),
                "home_logo": teams.get('home', {}).get('logo'),
                "away_logo": teams.get('away', {}).get('logo'),
                "home_code": teams.get('home', {}).get('code'),
                "away_code": teams.get('away', {}).get('code'),
                "league": league_info.get('name'),
                "country": league_info.get('country'),
                
                # Odds Mapping (Verify keys against Bzzoiro API response)
                "bookmaker": "Bet365", 
                "odds_1x2": odds.get('1x2', {}).get('home'),
                "odds_over25": odds.get('over_under', {}).get('over'),
                "odds_btts": odds.get('btts', {}).get('yes'),
                
                # CatBoost Model Outputs
                "prediction": prediction_results.get('prediction'),
                "confidence": prediction_results.get('confidence'),
                "home_win_prob": prediction_results.get('home_win_prob'),
                "draw_prob": prediction_results.get('draw_prob'),
                "away_win_prob": prediction_results.get('away_win_prob'),
                "tip": prediction_results.get('tip')
            }

            # 6. Push to Supabase (Triggers your new client logic)
            save_prediction(final_data)
            print(f"Saved: {final_data['home_team']} vs {final_data['away_team']}")

if __name__ == "__main__":
    run_daily_predictions()
            
