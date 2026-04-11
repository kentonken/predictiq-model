import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # We create a new dataframe for features
    feats = pd.DataFrame(index=df.index)

    # CRITICAL: If a column is missing, we shouldn't guess a 'middle' value.
    # We use NaN so the model knows the data is actually missing.
    def g(col):
        return df[col] if col in df.columns else np.nan

    # 1. Form & Results
    feats['home_win_rate_5'] = g('home_win_rate_5')
    feats['away_win_rate_5'] = g('away_win_rate_5')
    feats['home_elo'] = g('home_elo')
    feats['away_elo'] = g('away_elo')
    feats['elo_diff'] = feats['home_elo'] - feats['away_elo']
    
    # 2. Goals & xG (Removed the 1.4/1.1 hardcoded defaults)
    feats['home_xg_for_5'] = g('home_xg_for_5')
    feats['away_xg_for_5'] = g('away_xg_for_5')
    feats['poisson_home_goals'] = g('poisson_home_goals')
    feats['poisson_away_goals'] = g('poisson_away_goals')

    # ... Repeat for your other ~160 features ...
    # Ensure you convert everything to float at the end
    return feats.astype(float)
    
