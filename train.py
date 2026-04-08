import os
from api_bzzoiro import get_fixture
from predictor import predict_match, load_model
from supabase_client import save_prediction

def run_daily_predictions():
    model = load_model()
    # IDs: 2 (UCL), 183 (Kazakhstan), 31 (Azerbaijan), 10 (Iceland), 57 (Finland)
    leagues = [2, 183, 31, 10, 57] 
    
    for league_id in leagues:
        matches = get_fixture(league_id) 
        for match in matches:
            fixture = match.get('fixture', {})
            teams = match.get('teams', {})
            odds = match.get('odds', {}) 

            # THE TBD FIX: Precise splitting of the ISO date string
            raw_date = fixture.get('date') 
            m_date = raw_date.split('T')[0] if raw_date else None
            m_time = raw_date.split('T')[1][:5] if raw_date else None

            # Prediction logic
            prediction_results = predict_match(model, [], fixture.get('id'))

            final_data = {
                "fixture_id": fixture.get('id'),
                "match_date": m_date,
                "match_time": m_time,
                "home_team": teams.get('home', {}).get('name'),
                "away_team": teams.get('away', {}).get('name'),
                "home_logo": teams.get('home', {}).get('logo'),
                "away_logo": teams.get('away', {}).get('logo'),
                "league": match.get('league', {}).get('name'),
                "odds_1x2": odds.get('1x2', {}).get('home'),
                "prediction": prediction_results.get('prediction'),
                "confidence": prediction_results.get('confidence'),
                "tip": prediction_results.get('tip')
            }
            save_prediction(final_data)
            print(f"Sync Complete: {final_data['home_team']} vs {final_data['away_team']}")

if __name__ == "__main__":
    run_daily_predictions()
    
