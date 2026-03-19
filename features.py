from api_football import (
    get_fixture, get_team_last_home_fixtures,
    get_team_last_away_fixtures, get_head_to_head,
    get_standings, get_injuries,
)

FEATURE_COLUMNS = [
    "home_form_points", "away_form_points",
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "home_league_position", "away_league_position",
    "home_injuries_count", "away_injuries_count",
]

def _result_for_team(fixture, team_id):
    hg = fixture["goals"]["home"] or 0
    ag = fixture["goals"]["away"] or 0
    if hg == ag: return 1
    is_home = fixture["teams"]["home"]["id"] == team_id
    return 3 if (is_home and hg > ag) or (not is_home and ag > hg) else 0

def _goals_for_team(fixture, team_id):
    hg = fixture["goals"]["home"] or 0
    ag = fixture["goals"]["away"] or 0
    if fixture["teams"]["home"]["id"] == team_id:
        return hg, ag
    return ag, hg

def _form_points(fixtures, team_id):
    return sum(_result_for_team(f, team_id) for f in fixtures)

def _avg_goals(fixtures, team_id):
    if not fixtures: return 0.0, 0.0
    s, c = 0, 0
    for f in fixtures:
        gs, gc = _goals_for_team(f, team_id)
        s += gs; c += gc
    n = len(fixtures)
    return round(s/n, 2), round(c/n, 2)

def _h2h_record(h2h, home_id, away_id):
    hw = d = aw = 0
    for f in h2h:
        hg = f["goals"]["home"] or 0
        ag = f["goals"]["away"] or 0
        fhid = f["teams"]["home"]["id"]
        if hg == ag: d += 1
        elif hg > ag:
            if fhid == home_id: hw += 1
            else: aw += 1
        else:
            if fhid == away_id: aw += 1
            else: hw += 1
    return hw, d, aw

def _league_position(standings, team_id):
    for e in standings:
        if e["team"]["id"] == team_id: return e["rank"]
    return 20

def _count_injuries(injuries, team_id):
    return sum(1 for i in injuries if i["team"]["id"] == team_id
               and i["player"]["reason"] in ("Injury", "Suspension"))

def compute_features(fixture_id: int):
    fixture = get_fixture(fixture_id)
    if not fixture: return None
    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]
    league_id = fixture["league"]["id"]
    season = fixture["league"]["season"]
    hf = get_team_last_home_fixtures(home_id, 5)
    af = get_team_last_away_fixtures(away_id, 5)
    hs_avg, hc_avg = _avg_goals(hf, home_id)
    as_avg, ac_avg = _avg_goals(af, away_id)
    h2h = get_head_to_head(home_id, away_id, 5)
    hw, d, aw = _h2h_record(h2h, home_id, away_id)
    standings = get_standings(league_id, season)
    injuries = get_injuries(fixture_id)
    return {
        "fixture_id": fixture_id,
        "home_team": fixture["teams"]["home"]["name"],
        "away_team": fixture["teams"]["away"]["name"],
        "home_form_points": _form_points(hf, home_id),
        "away_form_points": _form_points(af, away_id),
        "home_goals_scored_avg": hs_avg,
        "home_goals_conceded_avg": hc_avg,
        "away_goals_scored_avg": as_avg,
        "away_goals_conceded_avg": ac_avg,
        "h2h_home_wins": hw, "h2h_draws": d, "h2h_away_wins": aw,
        "home_league_position": _league_position(standings, home_id),
        "away_league_position": _league_position(standings, away_id),
        "home_injuries_count": _count_injuries(injuries, home_id),
        "away_injuries_count": _count_injuries(injuries, away_id),
    }
