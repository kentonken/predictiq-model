from api_bzzoiro import get_fixture
from predictor import predict_match, load_model
from supabase_client import save_prediction

def run_daily_predictions():
    model = load_model()
    # Your niche leagues: Kazakhstan, Azerbaijan, Iceland, Finland, etc.
    leagues = [2, 183, 31, 10, 571]

    for league_id in leagues:
        matches = get_fixture(league_id)
        for match in matches:
            fix = match.get('fixture', {})
            tms = match.get('teams', {})
            
            # 1. FIX TBD: Extract components carefully
            raw_date = fix.get('date') # Expected: "2026-04-08T21:00:00+00:00"
            m_date = raw_date.split('T')[0] if raw_date else None
            m_time = raw_date.split('T')[1][:5] if raw_date else None

            # 2. FIX STATIC 38%: Extract real features from the match object
            # Replace these placeholders with actual API data fields
            home_rank = tms.get('home', {}).get('rank', 0)
            away_rank = tms.get('away', {}).get('rank', 0)
            is_cup = 1 if "Cup" in match.get('league', {}).get('name', '') else 0
            
            features = [home_rank, away_rank, is_cup] 

            # Get dynamic results based on actual features
            results = predict_match(model, features, fix.get('id'))

            payload = {
                "fixture_id": fix.get('id'),
                "match_date": m_date,
                "match_time": m_time,
                "home_team": tms.get('home', {}).get('name'),
                "away_team": tms.get('away', {}).get('name'),
                "home_logo": tms.get('home', {}).get('logo'),
                "away_logo": tms.get('away', {}).get('logo'),
                "prediction": results.get('prediction'),
                "confidence": results.get('confidence'),
                "tip": results.get('tip'),
                "league": match.get('league', {}).get('name')
            }

            save_prediction(payload)

if __name__ == "__main__":
    run_daily_predictions()
        
