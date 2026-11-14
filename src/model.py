"""
Phase 4: Train a model that forecasts home / draw / away probabilities and compare it
with the betting market.

Seasons are split by time, never shuffled, so the model is always tested on matches
that happened after everything it learned from:
  train      2021-22 to 2023-24   fit candidate models
  validate   2024-25              pick the best candidate
  test       2025-26 onwards      final, untouched comparison against the market
                                  (grows automatically as new matches are played)

The model never sees the bookmakers' odds: it uses team form and strength only, so the
comparison with the market is fair.
"""

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football.db"

# NumPy 2.0 on macOS raises false "overflow in matmul" warnings from Apple's maths library;
# the fitted coefficients and probabilities are finite and correct
warnings.filterwarnings("ignore", message=".*encountered in matmul")

TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24"]
VALID_SEASONS = ["2024-25"]

OUTCOMES = ["H", "D", "A"]
FEATURES = [
    "elo_diff",
    "points_last5_diff",
    "goals_for_last5_diff",
    "goals_against_last5_diff",
    "sot_for_last5_diff",
    "sot_against_last5_diff",
    "home_rest_days",
    "away_rest_days",
]
MARKET = ["mkt_home_prob", "mkt_draw_prob", "mkt_away_prob"]


def load_features(conn: sqlite3.Connection) -> pd.DataFrame:
    features = pd.read_sql("SELECT * FROM features", conn, parse_dates=["match_date"])
    return features.dropna(subset=FEATURES).reset_index(drop=True)


def candidate_models() -> dict:
    """Simple to complex. Logistic regression strength C: smaller = more cautious."""
    models = {
        f"Logistic regression (C={c})": make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=1000)
        )
        for c in [0.01, 0.1, 1.0]
    }
    models["Gradient boosting"] = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, random_state=42
    )
    return models


def predict_probs(model, X: pd.DataFrame) -> np.ndarray:
    """Probabilities in H, D, A order, whatever order the model stores its classes in."""
    probs = model.predict_proba(X)
    order = [list(model.classes_).index(o) for o in OUTCOMES]
    return probs[:, order]


def score(probs: np.ndarray, results: pd.Series) -> dict:
    """Log loss and Brier score: lower is better. Accuracy: share of favourites that won."""
    actual = np.column_stack([(results == o).to_numpy() for o in OUTCOMES]).astype(float)
    # Computed directly: sklearn's log_loss sorts labels alphabetically (A, D, H),
    # which would silently mismatch our H, D, A columns
    prob_of_result = np.clip(np.sum(probs * actual, axis=1), 1e-15, 1)
    return {
        "log_loss": -np.mean(np.log(prob_of_result)),
        "brier": np.mean(np.sum((probs - actual) ** 2, axis=1)),
        "accuracy": np.mean(np.array(OUTCOMES)[probs.argmax(axis=1)] == results.to_numpy()),
    }


def split(features: pd.DataFrame, seasons: list) -> pd.DataFrame:
    return features[features["season"].isin(seasons)]


def test_seasons(features: pd.DataFrame) -> list:
    """Every season after validation, so new seasons join the test set automatically."""
    return sorted(s for s in features["season"].unique() if s > VALID_SEASONS[-1])


def select_model(features: pd.DataFrame) -> tuple:
    train, valid = split(features, TRAIN_SEASONS), split(features, VALID_SEASONS)
    results = {}
    for name, model in candidate_models().items():
        model.fit(train[FEATURES], train["result"])
        results[name] = score(predict_probs(model, valid[FEATURES]), valid["result"])
    table = pd.DataFrame(results).T.sort_values("log_loss")
    return table.index[0], table


def base_rate_probs(train: pd.DataFrame, n: int) -> np.ndarray:
    """The no-skill forecast: give every match the training seasons' overall H/D/A rates."""
    rates = train["result"].value_counts(normalize=True)[OUTCOMES].to_numpy()
    return np.tile(rates, (n, 1))


def train_and_test(features: pd.DataFrame) -> tuple:
    best_name, validation = select_model(features)
    train = split(features, TRAIN_SEASONS + VALID_SEASONS)
    test = split(features, test_seasons(features)).copy()

    model = candidate_models()[best_name]
    model.fit(train[FEATURES], train["result"])
    model_probs = predict_probs(model, test[FEATURES])
    market_probs = test[MARKET].to_numpy()

    comparison = pd.DataFrame({
        "Base rates (no skill)": score(base_rate_probs(train, len(test)), test["result"]),
        f"Model: {best_name}": score(model_probs, test["result"]),
        "Betting market": score(market_probs, test["result"]),
    }).T

    test[["model_home_prob", "model_draw_prob", "model_away_prob"]] = model_probs
    return model, best_name, validation, comparison, test


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        features = load_features(conn)
        model, best_name, validation, comparison, test = train_and_test(features)

        predictions = test[[
            "match_id", "season", "match_date", "home_team", "away_team", "result",
            "avg_home_odds", "avg_draw_odds", "avg_away_odds",
            "model_home_prob", "model_draw_prob", "model_away_prob", *MARKET,
        ]].assign(match_date=lambda d: d["match_date"].dt.date.astype(str))
        predictions.to_sql("predictions", conn, if_exists="replace", index=False)

    pd.set_option("display.width", 120)
    print("Validation (2024-25), lower log loss is better:")
    print(validation.round(4), "\n")
    print(f"Chosen: {best_name}\n")
    print(f"Test ({', '.join(test_seasons(features))}, {len(test)} matches):")
    print(comparison.round(4), "\n")
    print("Saved test-set forecasts to the `predictions` table")


if __name__ == "__main__":
    main()
