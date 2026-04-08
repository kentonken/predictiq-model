from api_bzzoiro import get_fixture
from predictor import predict_match, load_model
from supabase_client import save_prediction

def run_daily_predictions():
    model = load_model()
    # Including Champions League (2) and niche leagues for accuracy
    leagues = [2, 183, 31, 10, 57] 
    
    for league_id in leagues:
        matches = get_fixture(league_id) 
        for match in matches:
            fix = match.get('fixture', {})
            tms = match.get('teams', {})
            
            # FIX TBD: Extract components from "2026-04-08T21:00:00+00:00"
            raw_date = fix.get('date') 
            m_date = raw_date.split('T')[0] if raw_date else None
            m_time = raw_date.split('T')[1][:5] if raw_date else None

            # Get professional prediction results
            # Note: Ensure you pass actual features here based on your model training
            results = predict_match(model, [1, 0, 1], fix.get('id')) 

            payload = {
                "fixture_id": str(fix.get('id')),
                "match_date": m_date,
                "match_time": m_time,
                "home_team": tms.get('home', {}).get('name'),
                "away_team": tms.get('away', {}).get('name'),
                "home_logo": tms.get('home', {}).get('logo'), 
                "away_logo": tms.get('away', {}).get('logo'),
                "prediction": results.get('prediction'), # "Home Win" or "Draw"
                "confidence": results.get('confidence'),
                "tip": results.get('tip'),
                "league": match.get('league', {}).get('name')
            }
            save_prediction(payload)
            
