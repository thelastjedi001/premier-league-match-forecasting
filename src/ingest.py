"""
Phase 1: Download Premier League results + bookmaker odds and load them into SQLite.

Run it as often as you like: each run replaces the seasons it downloads,
so the current season always has the latest results.
"""

import io
import sqlite3
from pathlib import Path

import pandas as pd
import requests

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football.db"
BASE_URL = "https://football-data.co.uk/mmz4281/{season}/{league}.csv"

LEAGUE = "E0"  # E0 = English Premier League
SEASONS = ["2122", "2223", "2324", "2425", "2526", "2627"]

# Raw column name -> clean column name
COLUMNS = {
    "Date": "match_date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",          # H = home win, D = draw, A = away win
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    "B365H": "b365_home_odds",  # Bet365 decimal odds
    "B365D": "b365_draw_odds",
    "B365A": "b365_away_odds",
    "AvgH": "avg_home_odds",    # average across bookmakers
    "AvgD": "avg_draw_odds",
    "AvgA": "avg_away_odds",
}


def download_season(season: str) -> pd.DataFrame:
    url = BASE_URL.format(season=season, league=LEAGUE)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.content.decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(text))


def clean(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    df = raw.dropna(subset=["HomeTeam", "AwayTeam"])  # drop blank trailing rows
    df = df[[c for c in COLUMNS if c in df.columns]].rename(columns=COLUMNS)
    df["match_date"] = pd.to_datetime(df["match_date"], dayfirst=True).dt.date.astype(str)
    df.insert(0, "season", f"20{season[:2]}-{season[2:]}")
    df.insert(1, "league", LEAGUE)
    return df


def load(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            season TEXT, league TEXT, match_date TEXT,
            home_team TEXT, away_team TEXT,
            home_goals INTEGER, away_goals INTEGER, result TEXT,
            home_shots INTEGER, away_shots INTEGER,
            home_shots_on_target INTEGER, away_shots_on_target INTEGER,
            home_corners INTEGER, away_corners INTEGER,
            home_yellows INTEGER, away_yellows INTEGER,
            home_reds INTEGER, away_reds INTEGER,
            b365_home_odds REAL, b365_draw_odds REAL, b365_away_odds REAL,
            avg_home_odds REAL, avg_draw_odds REAL, avg_away_odds REAL,
            PRIMARY KEY (league, match_date, home_team, away_team)
        )
    """)
    season = df["season"].iloc[0]
    conn.execute("DELETE FROM matches WHERE league = ? AND season = ?", (LEAGUE, season))
    df.to_sql("matches", conn, if_exists="append", index=False)


def main() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        for season in SEASONS:
            df = clean(download_season(season), season)
            load(df, conn)
            print(f"{df['season'].iloc[0]}: loaded {len(df)} matches")
        total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"Done. {total} matches in {DB_PATH}")


if __name__ == "__main__":
    main()
