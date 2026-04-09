"""
features.py — PredictIQ Pro Feature Engineering
163 features including Bzzoiro spatial data
Replaces existing features.py
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    def g(col, default):
        return df[col] if col in df.columns else default

    # ── 1. Form & Results (20) ─────────────────────────
    feats["home_win_rate_5"]      = g("home_win_rate_5", 0.5)
    feats["away_win_rate_5"]      = g("away_win_rate_5", 0.4)
    feats["home_draw_rate_5"]     = g("home_draw_rate_5", 0.2)
    feats["away_draw_rate_5"]     = g("away_draw_rate_5", 0.2)
    feats["home_loss_rate_5"]     = g("home_loss_rate_5", 0.3)
    feats["away_loss_rate_5"]     = g("away_loss_rate_5", 0.4)
    feats["home_form_pts_5"]      = g("home_form_pts_5", 9.0)
    feats["away_form_pts_5"]      = g("away_form_pts_5", 6.0)
    feats["home_win_rate_10"]     = g("home_win_rate_10", 0.5)
    feats["away_win_rate_10"]     = g("away_win_rate_10", 0.4)
    feats["h2h_home_wins"]        = g("h2h_home_wins", 3)
    feats["h2h_draws"]            = g("h2h_draws", 2)
    feats["h2h_away_wins"]        = g("h2h_away_wins", 1)
    feats["home_league_pos"]      = g("home_league_pos", 8)
    feats["away_league_pos"]      = g("away_league_pos", 12)
    feats["pos_diff"]             = feats["home_league_pos"] - feats["away_league_pos"]
    feats["home_elo"]             = g("home_elo", 1500)
    feats["away_elo"]             = g("away_elo", 1500)
    feats["elo_diff"]             = feats["home_elo"] - feats["away_elo"]
    feats["elo_home_win_prob"]    = 1 / (1 + 10 ** ((feats["away_elo"] - feats["home_elo"]) / 400))

    # ── 2. Goals & xG (25) ────────────────────────────
    feats["home_goals_scored_5"]   = g("home_goals_scored_5", 1.5)
    feats["away_goals_scored_5"]   = g("away_goals_scored_5", 1.1)
    feats["home_goals_conceded_5"] = g("home_goals_conceded_5", 1.0)
    feats["away_goals_conceded_5"] = g("away_goals_conceded_5", 1.4)
    feats["home_xg_for_5"]         = g("home_xg_for_5", 1.6)
    feats["away_xg_for_5"]         = g("away_xg_for_5", 1.2)
    feats["home_xg_against_5"]     = g("home_xg_against_5", 1.0)
    feats["away_xg_against_5"]     = g("away_xg_against_5", 1.5)
    feats["home_xg_diff"]          = feats["home_xg_for_5"] - feats["home_xg_against_5"]
    feats["away_xg_diff"]          = feats["away_xg_for_5"] - feats["away_xg_against_5"]
    feats["xg_matchup_home"]       = feats["home_xg_for_5"] * feats["away_xg_against_5"]
    feats["xg_matchup_away"]       = feats["away_xg_for_5"] * feats["home_xg_against_5"]
    feats["total_goals_avg_5"]     = feats["home_goals_scored_5"] + feats["away_goals_scored_5"]
    feats["home_btts_rate"]        = g("home_btts_rate", 0.5)
    feats["away_btts_rate"]        = g("away_btts_rate", 0.5)
    feats["h2h_avg_goals"]         = g("h2h_avg_goals", 2.5)
    feats["h2h_btts_rate"]         = g("h2h_btts_rate", 0.5)
    feats["home_over25_rate"]      = g("home_over25_rate", 0.5)
    feats["away_over25_rate"]      = g("away_over25_rate", 0.5)
    feats["home_clean_sheet_5"]    = g("home_clean_sheet_5", 0.3)
    feats["away_clean_sheet_5"]    = g("away_clean_sheet_5", 0.3)
    feats["home_scored_both_5"]    = g("home_scored_both_5", 0.6)
    feats["away_scored_both_5"]    = g("away_scored_both_5", 0.6)
    feats["poisson_home_goals"]    = g("poisson_home_goals", 1.4)
    feats["poisson_away_goals"]    = g("poisson_away_goals", 1.1)

    # ── 3. Shots (18) ─────────────────────────────────
    feats["home_shots_on_target_5"]  = g("home_shots_on_target_5", 4.5)
    feats["away_shots_on_target_5"]  = g("away_shots_on_target_5", 3.5)
    feats["home_total_shots_5"]      = g("home_total_shots_5", 12.0)
    feats["away_total_shots_5"]      = g("away_total_shots_5", 10.0)
    feats["home_shot_conversion_5"]  = g("home_shot_conversion_5", 0.12)
    feats["away_shot_conversion_5"]  = g("away_shot_conversion_5", 0.10)
    feats["home_xg_per_shot_5"]      = g("home_xg_per_shot_5", 0.11)
    feats["away_xg_per_shot_5"]      = g("away_xg_per_shot_5", 0.10)
    feats["home_shots_box_pct_5"]    = g("home_shots_box_pct_5", 0.55)
    feats["away_shots_box_pct_5"]    = g("away_shots_box_pct_5", 0.50)
    feats["home_shots_first_half_5"] = g("home_shots_first_half_5", 5.5)
    feats["away_shots_first_half_5"] = g("away_shots_first_half_5", 4.5)
    feats["home_shots_second_half_5"]= g("home_shots_second_half_5", 7.0)
    feats["away_shots_second_half_5"]= g("away_shots_second_half_5", 5.5)
    feats["home_post_hits_5"]        = g("home_post_hits_5", 0.8)
    feats["away_post_hits_5"]        = g("away_post_hits_5", 0.7)
    feats["home_penalty_rate"]       = g("home_penalty_rate", 0.08)
    feats["away_penalty_rate"]       = g("away_penalty_rate", 0.07)

    # ── 4. Spatial: Shotmap xG (12) — from Bzzoiro ────
    feats["home_avg_shot_x"]         = g("home_avg_shot_x", 85.0)
    feats["away_avg_shot_x"]         = g("away_avg_shot_x", 83.0)
    feats["home_avg_shot_xg"]        = g("home_avg_shot_xg", 0.12)
    feats["away_avg_shot_xg"]        = g("away_avg_shot_xg", 0.10)
    feats["home_xgot_ratio"]         = g("home_xgot_ratio", 1.1)
    feats["away_xgot_ratio"]         = g("away_xgot_ratio", 1.0)
    feats["home_big_chances_5"]      = g("home_big_chances_5", 2.5)
    feats["away_big_chances_5"]      = g("away_big_chances_5", 2.0)
    feats["home_header_goals_rate"]  = g("home_header_goals_rate", 0.20)
    feats["away_header_goals_rate"]  = g("away_header_goals_rate", 0.18)
    feats["home_set_piece_goals_rate"]= g("home_set_piece_goals_rate", 0.25)
    feats["away_set_piece_goals_rate"]= g("away_set_piece_goals_rate", 0.22)

    # ── 5. Spatial: Momentum (14) — from Bzzoiro ──────
    feats["home_momentum_avg_5"]     = g("home_momentum_avg_5", 5.0)
    feats["away_momentum_avg_5"]     = g("away_momentum_avg_5", -5.0)
    feats["home_momentum_peak_5"]    = g("home_momentum_peak_5", 25.0)
    feats["away_momentum_peak_5"]    = g("away_momentum_peak_5", 20.0)
    feats["home_momentum_variance_5"]= g("home_momentum_variance_5", 12.0)
    feats["away_momentum_variance_5"]= g("away_momentum_variance_5", 10.0)
    feats["home_pressure_index_5"]   = g("home_pressure_index_5", 0.55)
    feats["away_pressure_index_5"]   = g("away_pressure_index_5", 0.45)
    feats["home_late_pressure_5"]    = g("home_late_pressure_5", 0.60)
    feats["away_late_pressure_5"]    = g("away_late_pressure_5", 0.50)
    feats["home_comeback_rate"]      = g("home_comeback_rate", 0.20)
    feats["away_comeback_rate"]      = g("away_comeback_rate", 0.15)
    feats["home_goals_min_75_90_5"]  = g("home_goals_min_75_90_5", 0.4)
    feats["away_goals_min_75_90_5"]  = g("away_goals_min_75_90_5", 0.35)

    # ── 6. Spatial: Avg Positions / Tactics (16) ──────
    feats["home_avg_def_line_x"]     = g("home_avg_def_line_x", 35.0)
    feats["away_avg_def_line_x"]     = g("away_avg_def_line_x", 35.0)
    feats["home_avg_att_line_x"]     = g("home_avg_att_line_x", 75.0)
    feats["away_avg_att_line_x"]     = g("away_avg_att_line_x", 75.0)
    feats["home_team_width"]         = g("home_team_width", 55.0)
    feats["away_team_width"]         = g("away_team_width", 55.0)
    feats["home_team_compactness"]   = g("home_team_compactness", 40.0)
    feats["away_team_compactness"]   = g("away_team_compactness", 40.0)
    feats["home_pressing_intensity"] = g("home_pressing_intensity", 0.50)
    feats["away_pressing_intensity"] = g("away_pressing_intensity", 0.50)
    feats["home_possession_avg_5"]   = g("home_possession_avg_5", 52.0)
    feats["away_possession_avg_5"]   = g("away_possession_avg_5", 48.0)
    feats["home_passes_per_game_5"]  = g("home_passes_per_game_5", 450.0)
    feats["away_passes_per_game_5"]  = g("away_passes_per_game_5", 420.0)
    feats["home_crossing_rate_5"]    = g("home_crossing_rate_5", 0.25)
    feats["away_crossing_rate_5"]    = g("away_crossing_rate_5", 0.22)

    # ── 7. Defensive (15) ─────────────────────────────
    feats["home_tackles_won_5"]      = g("home_tackles_won_5", 18.0)
    feats["away_tackles_won_5"]      = g("away_tackles_won_5", 16.0)
    feats["home_interceptions_5"]    = g("home_interceptions_5", 12.0)
    feats["away_interceptions_5"]    = g("away_interceptions_5", 11.0)
    feats["home_aerials_won_5"]      = g("home_aerials_won_5", 22.0)
    feats["away_aerials_won_5"]      = g("away_aerials_won_5", 20.0)
    feats["home_fouls_conceded_5"]   = g("home_fouls_conceded_5", 11.0)
    feats["away_fouls_conceded_5"]   = g("away_fouls_conceded_5", 12.0)
    feats["home_yellow_cards_5"]     = g("home_yellow_cards_5", 1.5)
    feats["away_yellow_cards_5"]     = g("away_yellow_cards_5", 1.8)
    feats["home_red_cards_5"]        = g("home_red_cards_5", 0.1)
    feats["away_red_cards_5"]        = g("away_red_cards_5", 0.1)
    feats["home_corners_for_5"]      = g("home_corners_for_5", 5.5)
    feats["away_corners_for_5"]      = g("away_corners_for_5", 4.5)
    feats["corners_total_avg"]       = feats["home_corners_for_5"] + feats["away_corners_for_5"]

    # ── 8. Context (13) ───────────────────────────────
    feats["league_tier"]             = g("league_tier", 1)
    feats["match_week"]              = g("match_week", 20)
    feats["season_progress"]         = g("season_progress", 0.5)
    feats["is_derby"]                = g("is_derby", 0)
    feats["is_home_advantage"]       = g("is_home_advantage", 1)
    feats["home_squad_depth_score"]  = g("home_squad_depth_score", 0.7)
    feats["away_squad_depth_score"]  = g("away_squad_depth_score", 0.7)
    feats["home_injuries_count"]     = g("home_injuries_count", 2)
    feats["away_injuries_count"]     = g("away_injuries_count", 2)
    feats["home_prev_match_result"]  = g("home_prev_match_result", 1)
    feats["away_prev_match_result"]  = g("away_prev_match_result", 1)
    feats["home_season_goals_scored"]= g("home_season_goals_scored", 28)
    feats["away_season_goals_scored"]= g("away_season_goals_scored", 22)

    # ── 9. Bzzoiro pre-computed predictions as features (20) ──
    feats["bzz_prob_home_win"]       = g("bzz_prob_home_win", 45.0)
    feats["bzz_prob_draw"]           = g("bzz_prob_draw", 25.0)
    feats["bzz_prob_away_win"]       = g("bzz_prob_away_win", 30.0)
    feats["bzz_expected_home_goals"] = g("bzz_expected_home_goals", 1.4)
    feats["bzz_expected_away_goals"] = g("bzz_expected_away_goals", 1.1)
    feats["bzz_prob_over_15"]        = g("bzz_prob_over_15", 75.0)
    feats["bzz_prob_over_25"]        = g("bzz_prob_over_25", 55.0)
    feats["bzz_prob_over_35"]        = g("bzz_prob_over_35", 30.0)
    feats["bzz_prob_btts"]           = g("bzz_prob_btts", 50.0)
    feats["bzz_confidence"]          = g("bzz_confidence", 0.65)
    feats["bzz_home_win_recommend"]  = g("bzz_home_win_recommend", 0)
    feats["bzz_over_15_recommend"]   = g("bzz_over_15_recommend", 0)
    feats["bzz_over_25_recommend"]   = g("bzz_over_25_recommend", 0)
    feats["bzz_over_35_recommend"]   = g("bzz_over_35_recommend", 0)
    feats["bzz_btts_recommend"]      = g("bzz_btts_recommend", 0)
    feats["bzz_winner_recommend"]    = g("bzz_winner_recommend", 0)
    feats["bzz_home_xg_ratio"]       = feats["bzz_expected_home_goals"] / (feats["bzz_expected_home_goals"] + feats["bzz_expected_away_goals"] + 0.01)
    feats["bzz_total_xg"]            = feats["bzz_expected_home_goals"] + feats["bzz_expected_away_goals"]
    feats["bzz_goal_diff_expected"]  = feats["bzz_expected_home_goals"] - feats["bzz_expected_away_goals"]
    feats["bzz_model_version_enc"]   = g("bzz_model_version_enc", 5)

    return feats.astype(float)
    
