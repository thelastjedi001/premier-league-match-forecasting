"""
Phase 5: Backtest a betting strategy built on the model's forecasts.

The rule: bet a flat stake on any outcome where the model's probability times the
bookmakers' average odds promises a positive expected return ("edge") above a threshold.

The threshold is chosen on the 2024-25 validation season and then applied once to the
test seasons, so the test result is honest. A luck test asks how often a bettor would
make this much if the market's own probabilities were the truth.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import model as m

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football.db"

STAKE = 10                                   # £ per bet
EDGE_THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20]
MIN_BETS = 30                                # ignore thresholds with too few bets to judge
LUCK_SIMULATIONS = 10_000
BIG_SIX = {"Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"}

SIDES = {"H": "home", "D": "draw", "A": "away"}


def forecasts(features: pd.DataFrame, model_name: str, train_seasons: list, predict_seasons: list) -> pd.DataFrame:
    """Fit the chosen model on past seasons and forecast the seasons that follow."""
    train = m.split(features, train_seasons)
    target = m.split(features, predict_seasons).copy()
    model = m.candidate_models()[model_name]
    model.fit(train[m.FEATURES], train["result"])
    target[["model_home_prob", "model_draw_prob", "model_away_prob"]] = m.predict_probs(model, target[m.FEATURES])
    return target


def place_bets(matches: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    """One row per bet: every outcome whose expected return beats min_edge."""
    bets = []
    for outcome, side in SIDES.items():
        odds = matches[f"avg_{side}_odds"]
        edge = matches[f"model_{side}_prob"] * odds - 1
        chosen = matches[edge > min_edge]
        won = chosen["result"] == outcome
        bets.append(pd.DataFrame({
            "match_id": chosen["match_id"],
            "season": chosen["season"],
            "match_date": chosen["match_date"],
            "home_team": chosen["home_team"],
            "away_team": chosen["away_team"],
            "bet_on": outcome,
            "model_prob": chosen[f"model_{side}_prob"],
            "market_prob": chosen[f"mkt_{side}_prob"],
            "odds": odds[chosen.index],
            "edge": edge[chosen.index],
            "result": chosen["result"],
            "won": won,
            "profit": np.where(won, STAKE * (odds[chosen.index] - 1), -STAKE),
        }))
    return pd.concat(bets).sort_values(["match_date", "match_id"]).reset_index(drop=True)


def summarise(bets: pd.DataFrame) -> dict:
    staked = STAKE * len(bets)
    return {
        "bets": len(bets),
        "win_rate": bets["won"].mean() if len(bets) else np.nan,
        "avg_odds": bets["odds"].mean() if len(bets) else np.nan,
        "profit": bets["profit"].sum(),
        "roi": bets["profit"].sum() / staked if staked else np.nan,
    }


def breakdown(bets: pd.DataFrame, by: str) -> pd.DataFrame:
    return pd.DataFrame({key: summarise(group) for key, group in bets.groupby(by)}).T.rename_axis(by)


def threshold_table(matches: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({t: summarise(place_bets(matches, t)) for t in EDGE_THRESHOLDS}).T.rename_axis("min_edge")


def choose_threshold(validation_table: pd.DataFrame) -> float:
    enough = validation_table[validation_table["bets"] >= MIN_BETS]
    return enough["roi"].astype(float).idxmax()


def luck_test(bets: pd.DataFrame, seed: int = 42) -> float:
    """Share of simulated seasons, with outcomes drawn from the market's probabilities,
    that make at least as much profit as the strategy really did."""
    rng = np.random.default_rng(seed)
    wins = rng.random((LUCK_SIMULATIONS, len(bets))) < bets["market_prob"].to_numpy()
    payout = np.where(wins, STAKE * (bets["odds"].to_numpy() - 1), -STAKE)
    return float(np.mean(payout.sum(axis=1) >= bets["profit"].sum()))


def benchmark(matches: pd.DataFrame, pick) -> dict:
    """A naive strategy that bets on one outcome per match, chosen by `pick`."""
    picks = matches.apply(pick, axis=1)
    odds = np.array([row[f"avg_{SIDES[p]}_odds"] for (_, row), p in zip(matches.iterrows(), picks)])
    won = (matches["result"] == picks).to_numpy()
    return summarise(pd.DataFrame({"won": won, "odds": odds, "profit": np.where(won, STAKE * (odds - 1), -STAKE)}))


def run() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        features = m.load_features(conn)
    model_name, _ = m.select_model(features)

    valid = forecasts(features, model_name, m.TRAIN_SEASONS, m.VALID_SEASONS)
    test = forecasts(features, model_name, m.TRAIN_SEASONS + m.VALID_SEASONS, m.test_seasons(features))

    validation_table = threshold_table(valid)
    threshold = choose_threshold(validation_table)
    bets = place_bets(test, threshold)
    bets["cumulative_profit"] = bets["profit"].cumsum()
    bets["against_big_six"] = np.where(bets["bet_on"] == "H", bets["away_team"], bets["home_team"]).astype(object)
    bets["against_big_six"] = bets["against_big_six"].isin(BIG_SIX) & (bets["bet_on"] != "D")

    return {
        "model_name": model_name,
        "validation_table": validation_table,
        "threshold": threshold,
        "test_table": threshold_table(test),
        "bets": bets,
        "summary": summarise(bets),
        "luck_p_value": luck_test(bets),
        "benchmarks": pd.DataFrame({
            "Always bet home": benchmark(test, lambda r: "H"),
            "Always bet the market favourite": benchmark(
                test, lambda r: ["H", "D", "A"][int(np.argmax([r.mkt_home_prob, r.mkt_draw_prob, r.mkt_away_prob]))]),
            "Our strategy": summarise(bets),
        }).T,
        "test": test,
    }


def main() -> None:
    results = run()
    bets = results["bets"]
    with sqlite3.connect(DB_PATH) as conn:
        bets.assign(match_date=bets["match_date"].dt.date.astype(str)).to_sql(
            "bets", conn, if_exists="replace", index=False)

    pd.set_option("display.width", 120)
    print("Validation (2024-25) by minimum edge:")
    print(results["validation_table"].round(3), "\n")
    print(f"Chosen minimum edge: {results['threshold']:.0%}\n")
    print("Test seasons, chosen strategy vs naive benchmarks:")
    print(results["benchmarks"].round(3), "\n")
    print(f"Luck test: {results['luck_p_value']:.1%} of market-driven simulations made at least as much\n")
    print("Saved every test-season bet to the `bets` table")


if __name__ == "__main__":
    main()
