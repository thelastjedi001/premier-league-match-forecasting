"""
Phase 3: Build pre-match features for every match and save them to the `features` table.

Every feature uses only information available before kick-off (no data leakage):
rolling stats are shifted by one game, and Elo ratings are recorded before they are
updated with the match result.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football.db"

WINDOW = 5                # rolling window: a team's last 5 matches
MIN_GAMES = 3             # need at least 3 previous matches for a rolling value
NEW_SPELL_GAP_DAYS = 150  # a longer gap than a summer break means the team was relegated
MAX_REST_DAYS = 14        # anything over two weeks counts as fully rested

ELO_START = 1500          # every team in the first season
ELO_PROMOTED = 1450       # promoted teams start below the league average
ELO_K = 20                # how far one result moves a rating
ELO_HOME_ADVANTAGE = 60   # rating points added to the home team when predicting
ELO_SEASON_REGRESSION = 0.2  # pull ratings 20% back toward 1500 each summer

POINTS = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}  # result -> (home points, away points)
ROLLING_STATS = ["points", "goals_for", "goals_against", "sot_for", "sot_against"]


def number_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Sort by date and give every match an id in that order."""
    matches = matches.drop(columns="match_id", errors="ignore")
    matches = matches.sort_values(["match_date", "home_team"]).reset_index(drop=True)
    matches.insert(0, "match_id", matches.index)
    return matches


def load_matches(conn: sqlite3.Connection) -> pd.DataFrame:
    return number_matches(pd.read_sql("SELECT * FROM matches", conn, parse_dates=["match_date"]))


def team_view(matches: pd.DataFrame) -> pd.DataFrame:
    """Turn one row per match into two rows per match: one from each team's point of view."""
    sides = []
    for side, other, is_home, pts in [("home", "away", 1, 0), ("away", "home", 0, 1)]:
        sides.append(pd.DataFrame({
            "match_id": matches["match_id"],
            "match_date": matches["match_date"],
            "team": matches[f"{side}_team"],
            "is_home": is_home,
            "points": matches["result"].map({r: p[pts] for r, p in POINTS.items()}),  # blank if unplayed
            "goals_for": matches[f"{side}_goals"],
            "goals_against": matches[f"{other}_goals"],
            "sot_for": matches[f"{side}_shots_on_target"],
            "sot_against": matches[f"{other}_shots_on_target"],
        }))
    return pd.concat(sides).sort_values(["team", "match_date"]).reset_index(drop=True)


def add_rolling_stats(teams: pd.DataFrame) -> pd.DataFrame:
    """Average of each stat over the team's previous WINDOW matches, excluding this one."""
    gap_days = teams.groupby("team")["match_date"].diff().dt.days

    # A promoted team's last Premier League games may be years old, so start its history again
    teams["spell"] = (gap_days.isna() | (gap_days > NEW_SPELL_GAP_DAYS)).cumsum()
    by_spell = teams.groupby("spell")

    for stat in ROLLING_STATS:
        teams[f"{stat}_last{WINDOW}"] = by_spell[stat].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()
        )

    teams["rest_days"] = gap_days.clip(upper=MAX_REST_DAYS).fillna(MAX_REST_DAYS)
    return teams


def elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """Each team's Elo rating going into every match, updated after each result."""
    ratings = {}
    first_season = matches["season"].min()
    season, last_season_teams, this_season_teams = None, set(), set()
    rows = []

    for m in matches.itertuples():
        if m.season != season:
            if season is not None:
                ratings = {t: r + ELO_SEASON_REGRESSION * (ELO_START - r) for t, r in ratings.items()}
                last_season_teams = this_season_teams
            season, this_season_teams = m.season, set()

        for team in (m.home_team, m.away_team):
            if team not in this_season_teams:
                this_season_teams.add(team)
                if m.season == first_season:
                    ratings[team] = ELO_START
                elif team not in last_season_teams:
                    ratings[team] = ELO_PROMOTED

        home, away = ratings[m.home_team], ratings[m.away_team]
        rows.append((m.match_id, home, away))
        if pd.isna(m.result):  # an upcoming fixture: rate it, but nothing to learn yet
            continue

        expected_home = 1 / (1 + 10 ** ((away - (home + ELO_HOME_ADVANTAGE)) / 400))
        actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[m.result]
        change = ELO_K * (actual_home - expected_home)
        ratings[m.home_team] = home + change
        ratings[m.away_team] = away - change

    return pd.DataFrame(rows, columns=["match_id", "home_elo", "away_elo"])


def market_probabilities(matches: pd.DataFrame) -> pd.DataFrame:
    """Bookmaker probabilities with the margin removed, so the three outcomes sum to 1."""
    raw = 1 / matches[["avg_home_odds", "avg_draw_odds", "avg_away_odds"]]
    fair = raw.div(raw.sum(axis=1), axis=0)
    fair.columns = ["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]
    return pd.concat([matches[["match_id"]], fair], axis=1)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    teams = add_rolling_stats(team_view(matches))
    team_cols = [f"{s}_last{WINDOW}" for s in ROLLING_STATS] + ["rest_days"]

    features = matches[[
        "match_id", "season", "match_date", "home_team", "away_team",
        "result", "home_goals", "away_goals",
        "avg_home_odds", "avg_draw_odds", "avg_away_odds",
    ]]
    for side, is_home in [("home", 1), ("away", 0)]:
        side_stats = teams.loc[teams["is_home"] == is_home, ["match_id"] + team_cols]
        features = features.merge(side_stats.add_prefix(f"{side}_").rename(
            columns={f"{side}_match_id": "match_id"}), on="match_id")

    features = features.merge(elo_ratings(matches), on="match_id")
    features = features.merge(market_probabilities(matches), on="match_id")

    # Home-minus-away differences: often the most useful single signals
    features["elo_diff"] = features["home_elo"] - features["away_elo"]
    for stat in ROLLING_STATS:
        col = f"{stat}_last{WINDOW}"
        features[f"{col}_diff"] = features[f"home_{col}"] - features[f"away_{col}"]

    return features.sort_values("match_id").reset_index(drop=True)


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        features = build_features(load_matches(conn))
        out = features.assign(match_date=features["match_date"].dt.date.astype(str))
        out.to_sql("features", conn, if_exists="replace", index=False)

    complete = features.dropna().shape[0]
    print(f"Built {features.shape[1]} columns for {len(features)} matches")
    print(f"{complete} matches have every feature (the rest are early-season or newly promoted teams)")
    print(f"Saved to the `features` table in {DB_PATH}")


if __name__ == "__main__":
    main()
