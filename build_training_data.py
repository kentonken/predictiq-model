import os, time, argparse, requests
import pandas as pd
from dotenv import load_dotenv
from features import compute_features

load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
OUTPUT_PATH = "data/historical_matches.csv"

def get_finished_fixtures(league_id, season):
    print(f"Fetching fixtures for league={league_id}, season={season}...")
    res = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS,
        params={"league": league_id, "season": season, "status": "FT"}, timeout=15)
    res.raise_for_status()
    fixtures = res.json().get("response", [])
    print(f"  Found {len(fixtures)} finished fixtures.")
    return fixtures

def result_label(fixture):
    hg = fixture["goals"]["home"] or 0
    ag = fixture["goals"]["away"] or 0
    if hg > ag: return 0
    elif hg == ag: return 1
    else: return 2

def build_dataset(league_id, season, max_fixtures=200):
    fixtures = get_finished_fixtures(league_id, season)[:max_fixtures]
    rows = []
    for i, fix in enumerate(fixtures):
        fid = fix["fixture"]["id"]
        print(f"  [{i+1}/{len(fixtures)}] Processing fixture {fid}...")
        try:
            features = compute_features(fid)
            if features is None:
                print(f"    Skipped")
                continue
            features["result"] = result_label(fix)
            rows.append(features)
            time.sleep(3)
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(2)
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=int, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--max", type=int, default=200)
    args = parser.parse_args()
    os.makedirs("data", exist_ok=True)
    rows = build_dataset(args.league, args.season, args.max)
    if not rows:
        print("No data collected.")
        return
    df_new = pd.DataFrame(rows)
    if os.path.exists(OUTPUT_PATH):
        df = pd.concat([pd.read_csv(OUTPUT_PATH), df_new], ignore_index=True)
        df = df.drop_duplicates(subset="fixture_id")
    else:
        df = df_new
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} matches to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
