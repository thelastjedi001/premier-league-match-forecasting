"""
Phase 6: Forecast the next Premier League fixtures.

Downloads upcoming fixtures and odds, builds the same pre-match features as for past
matches, trains the chosen model on every completed match, and writes the model's and
the market's probabilities to outputs/upcoming_predictions.csv.

The edge columns show where the model disagrees with the odds. The Phase 5 backtest found
no reliable profit from betting on those disagreements, so they are for analysis, not tips.
"""

import io
import sqlite3
from pathlib import Path

import pandas as pd
import requests

import model as m
from features import build_features, load_matches, number_matches
from ingest import LEAGUE, season_label

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "football.db"
OUTPUT_PATH = ROOT / "outputs" / "upcoming_predictions.csv"
FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"

FIXTURE_COLUMNS = {
    "Date": "match_date",
    "Time": "kickoff",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "AvgH": "avg_home_odds",
    "AvgD": "avg_draw_odds",
    "AvgA": "avg_away_odds",
}
OUTPUT_COLUMNS = [
    "match_date", "kickoff", "home_team", "away_team",
    "model_home_prob", "model_draw_prob", "model_away_prob",
    "mkt_home_prob", "mkt_draw_prob", "mkt_away_prob",
    "avg_home_odds", "avg_draw_odds", "avg_away_odds",
    "home_edge", "draw_edge", "away_edge",
]


def download_fixtures() -> pd.DataFrame:
    response = requests.get(FIXTURES_URL, timeout=30)
    response.raise_for_status()
    raw = pd.read_csv(io.StringIO(response.content.decode("utf-8-sig", errors="replace")))
    fixtures = raw[raw["Div"] == LEAGUE][list(FIXTURE_COLUMNS)].rename(columns=FIXTURE_COLUMNS)
    fixtures["match_date"] = pd.to_datetime(fixtures["match_date"], dayfirst=True)
    return fixtures.dropna(subset=["avg_home_odds", "avg_draw_odds", "avg_away_odds"])


def forecast_fixtures(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Probabilities for each fixture from a model trained on every completed match."""
    if fixtures.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    upcoming = fixtures.assign(
        season=fixtures["match_date"].dt.date.map(season_label),
        upcoming=True,
    )
    combined = number_matches(pd.concat([matches.assign(upcoming=False), upcoming], ignore_index=True))
    features = build_features(combined).merge(combined[["match_id", "upcoming", "kickoff"]], on="match_id")

    played = features[~features["upcoming"]].dropna(subset=m.FEATURES)
    to_forecast = features[features["upcoming"]].copy()

    model_name, _ = m.select_model(played)
    model = m.candidate_models()[model_name]
    model.fit(played[m.FEATURES], played["result"])

    # A newly promoted team has no recent form yet: fill with the league's typical value
    filled = to_forecast[m.FEATURES].fillna(played[m.FEATURES].median())
    to_forecast[["model_home_prob", "model_draw_prob", "model_away_prob"]] = m.predict_probs(model, filled)

    for side in ["home", "draw", "away"]:
        to_forecast[f"{side}_edge"] = to_forecast[f"model_{side}_prob"] * to_forecast[f"avg_{side}_odds"] - 1

    to_forecast["match_date"] = to_forecast["match_date"].dt.date.astype(str)
    return to_forecast[OUTPUT_COLUMNS].reset_index(drop=True)


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        matches = load_matches(conn)
    forecasts = forecast_fixtures(matches, download_fixtures())

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    forecasts.round(4).to_csv(OUTPUT_PATH, index=False)
    if forecasts.empty:
        print("No Premier League fixtures with odds published yet (international break or off-season)")
    else:
        print(f"Forecast {len(forecasts)} upcoming fixtures")
        print(forecasts[["match_date", "home_team", "away_team",
                         "model_home_prob", "model_draw_prob", "model_away_prob"]].round(3).to_string(index=False))
    print(f"Saved to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
