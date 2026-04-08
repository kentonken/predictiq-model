-- 1. Create Players Table
CREATE TABLE IF NOT EXISTS players (
    id BIGINT PRIMARY KEY,
    team_id BIGINT,
    name TEXT,
    position TEXT,
    rating FLOAT,
    photo_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Create Player Match Stats (for Heatmaps)
CREATE TABLE IF NOT EXISTS player_match_stats (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    player_id BIGINT REFERENCES players(id),
    fixture_id TEXT,
    heatmap_data JSONB, -- Stores the touch coordinates
    metrics JSONB,      -- Stores the 30+ metrics
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Enhance existing predictions table
ALTER TABLE ai_predictions 
ADD COLUMN IF NOT EXISTS bzzoiro_raw_ml JSONB,
ADD COLUMN IF NOT EXISTS most_likely_score TEXT,
ADD COLUMN IF NOT EXISTS bzzoiro_confidence FLOAT;
