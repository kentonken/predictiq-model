import os
import time
import requests
from supabase import create_client, Client

# Configuration (Railway Variables)
API_KEY      = os.getenv("BZZOIRD_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Bzzoiro base
BASE_URL = "https://sports.bzzoiro.com/api"
HEADERS  = {
    "Authorization": f"Token {API_KEY}",
    "Accept": "application/json"
}

# Lazy Supabase client — only created when actually needed
_supabase_client = None

def _get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# ─────────────────────────────────────────────
# SPATIAL DATA HELPERS
# ─────────────────────────────────────────────

def _fetch_event_spatial(event_id: int) -> dict:
    """Fetch shotmap + momentum + avg_positions from a finished match."""
    try:
        r = requests.get(
            f"{BASE_URL}/events/{event_id}/",
            headers=HEADERS, timeout=30
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  Spatial fetch failed event {event_id}: {e}")
        return {}


def _aggregate_spatial(spatial: dict) -> dict:
    """Flatten shotmap / momentum / avg_positions into flat feature dict."""
    out = {}
    shotmap  = spatial.get("shotmap", [])
    momentum = spatial.get("momentum", [])
    avg_pos  = spatial.get("average_positions", {})

    # Shotmap per side
    for side, is_home in [("home", True), ("away", False)]:
        shots = [s for s in shotmap if s.get("home") == is_home]
        if shots:
            xs    = [s["pos"]["x"] for s in shots if "pos" in s]
            xgs   = [s.get("xg", 0) for s in shots]
            xgots = [s.get("xgot", 0) for s in shots if s.get("xgot")]
            out[f"{side}_avg_shot_x"]           = sum(xs) / len(xs) if xs else 85
            out[f"{side}_avg_shot_xg"]          = sum(xgs) / len(xgs) if xgs else 0.10
            out[f"{side}_xgot_ratio"]           = sum(xgots) / sum(xgs) if sum(xgs) > 0 else 1.0
            out[f"{side}_header_goals_rate"]    = sum(1 for s in shots if s.get("body") == "head") / len(shots)
            out[f"{side}_set_piece_goals_rate"] = sum(1 for s in shots if s.get("sit") in ["corner", "free-kick", "penalty"]) / len(shots)

    # Momentum
    if momentum:
        vals   = [m["value"] for m in momentum]
        late   = [m["value"] for m in momentum if m.get("minute", 0) >= 75]
        home_v = [v for v in vals if v > 0]
        away_v = [abs(v) for v in vals if v < 0]
        out["home_momentum_avg_5"]  = sum(home_v) / len(home_v) if home_v else 0
        out["away_momentum_avg_5"]  = sum(away_v) / len(away_v) if away_v else 0
        out["home_momentum_peak_5"] = max(home_v) if home_v else 0
        out["away_momentum_peak_5"] = max(away_v) if away_v else 0
        out["home_late_pressure_5"] = sum(v for v in late if v > 0) / (len(late) + 1)
        out["away_late_pressure_5"] = abs(sum(v for v in late if v < 0)) / (len(late) + 1)

    # Avg positions
    for side in ["home", "away"]:
        players = avg_pos.get(side, [])
        if players:
            xs = sorted([p["pos"]["x"] for p in players if "pos" in p])
            ys = [p["pos"]["y"] for p in players if "pos" in p]
            if xs:
                out[f"{side}_avg_def_line_x"]   = xs[2] if len(xs) > 4 else xs[0]
                out[f"{side}_avg_att_line_x"]   = xs[-3] if len(xs) > 3 else xs[-1]
                out[f"{side}_team_compactness"] = xs[-1] - xs[0]
            if ys:
                out[f"{side}_team_width"] = max(ys) - min(ys)

    return out


# ─────────────────────────────────────────────
# TRAINING DATA FETCH
# ─────────────────────────────────────────────

def get_finished_predictions_for_training(
    date_from: str,
    date_to: str,
    league_id: int = None,
    sleep_ms: int = 300,
) -> list:
    """
    Fetch finished match predictions + spatial data from Bzzoiro.
    Returns list of flat dicts ready for engineer_features().
    """
    params = {"upcoming": "false", "date_from": date_from, "date_to": date_to}
    if league_id:
        params["league"] = league_id

    r = requests.get(
        f"{BASE_URL}/predictions/",
        headers=HEADERS,
        params=params,
        timeout=30
    )
    r.raise_for_status()
    predictions = r.json().get("results", [])
    print(f"  {len(predictions)} finished predictions found")

    rows = []
    for i, pred in enumerate(predictions):
        ev = pred["event"]

        # Determine actual result
        hs  = ev.get("home_score")
        as_ = ev.get("away_score")
        if hs is None or as_ is None:
            continue
        result = "H" if hs > as_ else ("A" if as_ > hs else "D")

        row = {
            "result":                  result,
            "league_id":               ev["league"]["id"],
            "bzz_prob_home_win":       pred.get("prob_home_win", 45),
            "bzz_prob_draw":           pred.get("prob_draw", 25),
            "bzz_prob_away_win":       pred.get("prob_away_win", 30),
            "bzz_expected_home_goals": pred.get("expected_home_goals", 1.4),
            "bzz_expected_away_goals": pred.get("expected_away_goals", 1.1),
            "bzz_prob_over_15":        pred.get("prob_over_15", 75),
            "bzz_prob_over_25":        pred.get("prob_over_25", 55),
            "bzz_prob_over_35":        pred.get("prob_over_35", 30),
            "bzz_prob_btts":           pred.get("prob_btts_yes", 50),
            "bzz_confidence":          pred.get("confidence", 0.65),
            "bzz_home_win_recommend":  int(pred.get("favorite_recommend", False)),
            "bzz_over_15_recommend":   int(pred.get("over_15_recommend", False)),
            "bzz_over_25_recommend":   int(pred.get("over_25_recommend", False)),
            "bzz_over_35_recommend":   int(pred.get("over_35_recommend", False)),
            "bzz_btts_recommend":      int(pred.get("btts_recommend", False)),
            "bzz_winner_recommend":    int(pred.get("winner_recommend", False)),
            "bzz_model_version_enc":   5,
            "poisson_home_goals":      pred.get("expected_home_goals", 1.4),
            "poisson_away_goals":      pred.get("expected_away_goals", 1.1),
        }

        # Enrich with spatial data
        spatial = _fetch_event_spatial(ev["id"])
        row.update(_aggregate_spatial(spatial))

        rows.append(row)
        print(f"  [{i+1}/{len(predictions)}] event {ev['id']} → {result}")
        time.sleep(sleep_ms / 1000)

    return rows


# ─────────────────────────────────────────────
# LIVE PREDICTIONS (used by Supabase edge fn)
# ─────────────────────────────────────────────

def get_upcoming_predictions(league_id: int = None) -> list:
    """Fetch today's upcoming predictions from Bzzoiro."""
    params = {"upcoming": "true"}
    if league_id:
        params["league"] = league_id
    r = requests.get(f"{BASE_URL}/predictions/", headers=HEADERS,
                     params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def sync_predictions_to_supabase(predictions: list) -> int:
    """Upsert Bzzoiro predictions into Supabase predictions table."""
    sb = _get_supabase()
    count = 0
    for pred in predictions:
        ev = pred["event"]
        row = {
            "bzzoiro_event_id":   ev["id"],
            "home_team":          ev["home_team"],
            "away_team":          ev["away_team"],
            "event_date":         ev["event_date"],
            "league_name":        ev["league"]["name"],
            "prob_home_win":      pred.get("prob_home_win"),
            "prob_draw":          pred.get("prob_draw"),
            "prob_away_win":      pred.get("prob_away_win"),
            "expected_home_goals":pred.get("expected_home_goals"),
            "expected_away_goals":pred.get("expected_away_goals"),
            "prob_over_25":       pred.get("prob_over_25"),
            "prob_btts":          pred.get("prob_btts_yes"),
            "confidence":         pred.get("confidence"),
            "model_version":      pred.get("model_version", ""),
        }
        try:
            sb.table("predictions").upsert(row, on_conflict="bzzoiro_event_id").execute()
            count += 1
        except Exception as e:
            print(f"  Upsert failed event {ev['id']}: {e}")
    return count


if __name__ == "__main__":
    # Quick test
    data = get_finished_predictions_for_training("2026-01-01", "2026-01-31")
    print(f"Fetched {len(data)} rows")
        
