"""
Unit tests on small made-up seasons, so they run in seconds without downloading anything.
GitHub Actions runs them on every push and before every weekly update.
"""

from datetime import date
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from features import build_features, number_matches
from ingest import season_label, seasons_to_download
from model import score

TEAMS = ["Ashford", "Brook", "Carlton", "Dale"]


@pytest.fixture
def matches() -> pd.DataFrame:
    """Four teams playing everyone home and away, one match a week, made-up results."""
    rng = np.random.default_rng(0)
    rows = []
    for week, (home, away) in enumerate(list(permutations(TEAMS, 2)) * 2):
        home_goals, away_goals = rng.integers(0, 4, size=2)
        rows.append({
            "season": "2024-25",
            "match_date": pd.Timestamp("2024-08-10") + pd.Timedelta(weeks=week),
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D",
            "home_shots_on_target": home_goals + 2,
            "away_shots_on_target": away_goals + 2,
            "avg_home_odds": 2.2,
            "avg_draw_odds": 3.4,
            "avg_away_odds": 3.3,
        })
    return number_matches(pd.DataFrame(rows))


def team_points(matches: pd.DataFrame, team: str) -> pd.Series:
    home = matches["home_team"] == team
    points = np.select(
        [matches["result"] == "D", home & (matches["result"] == "H"), ~home & (matches["result"] == "A")],
        [1, 3, 3], 0)
    played = home | (matches["away_team"] == team)
    return pd.Series(points, index=matches.index)[played]


def test_rolling_form_uses_only_previous_matches(matches):
    features = build_features(matches)
    points = team_points(matches, "Ashford")
    tenth = points.index[9]
    expected = points.iloc[4:9].mean()  # the five games before Ashford's 10th
    row = features.loc[features["match_id"] == matches.loc[tenth, "match_id"]].iloc[0]
    side = "home" if row["home_team"] == "Ashford" else "away"
    assert row[f"{side}_points_last5"] == pytest.approx(expected)


def test_no_leakage_from_a_matchs_own_result(matches):
    target = 12
    tampered = matches.copy()
    flipped = [9, 0, "H"] if matches.loc[target, "result"] != "H" else [0, 9, "A"]
    tampered.loc[target, ["home_goals", "away_goals", "result"]] = flipped

    before, after = build_features(matches), build_features(tampered)
    pre_match = [c for c in before.columns if c.endswith(("_last5", "_elo", "_diff", "_prob"))]

    assert before.loc[target, pre_match].equals(after.loc[target, pre_match])
    assert not before.loc[target + 1:, "elo_diff"].equals(after.loc[target + 1:, "elo_diff"])


def test_upcoming_fixture_gets_features_without_a_result(matches):
    fixture = pd.DataFrame([{
        "season": "2024-25", "match_date": matches["match_date"].max() + pd.Timedelta(weeks=1),
        "home_team": "Ashford", "away_team": "Brook",
        "avg_home_odds": 2.0, "avg_draw_odds": 3.5, "avg_away_odds": 3.8,
    }])
    features = build_features(number_matches(pd.concat([matches, fixture], ignore_index=True)))
    upcoming = features.iloc[-1]
    assert pd.isna(upcoming["result"])
    assert upcoming[["home_elo", "away_elo", "home_points_last5", "away_points_last5"]].notna().all()


def test_market_probabilities_remove_the_margin(matches):
    features = build_features(matches)
    totals = features[["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]].sum(axis=1)
    assert np.allclose(totals, 1)


def test_log_loss_reads_columns_in_home_draw_away_order():
    # Confident and right twice: log loss must be small. sklearn's log_loss sorts the labels
    # alphabetically (A, D, H) and gets this badly wrong, which is why model.score avoids it.
    probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]])
    assert score(probs, pd.Series(["H", "A"]))["log_loss"] == pytest.approx(-np.log(0.9))


def test_seasons_roll_over_in_august():
    assert season_label(date(2026, 11, 14)) == "2026-27"
    assert season_label(date(2027, 5, 1)) == "2026-27"
    assert season_label(date(2027, 8, 1)) == "2027-28"
    assert seasons_to_download(date(2027, 8, 20))[-2:] == ["2627", "2728"]
