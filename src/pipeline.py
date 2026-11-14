"""
Phase 6: Run the whole project end to end with one command.

    python src/pipeline.py

Downloads the latest results, rebuilds features, retrains and re-tests the model, re-runs
the backtest, forecasts upcoming fixtures, and writes a summary to
outputs/model_performance.json. GitHub Actions runs this every week.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import backtest
import features
import ingest
import model
import predict

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "football.db"
SUMMARY_PATH = ROOT / "outputs" / "model_performance.json"

STEPS = [
    ("Download results", ingest.main),
    ("Build features", features.main),
    ("Train and test the model", model.main),
    ("Backtest the betting strategy", backtest.main),
    ("Forecast upcoming fixtures", predict.main),
]


def rounded(scores: dict) -> dict:
    return {k: round(float(v), 4) for k, v in scores.items()}


def summary(conn: sqlite3.Connection) -> dict:
    predictions = pd.read_sql("SELECT * FROM predictions", conn)
    bets = pd.read_sql("SELECT * FROM bets", conn)
    upcoming = pd.read_csv(predict.OUTPUT_PATH)
    model_probs = predictions[["model_home_prob", "model_draw_prob", "model_away_prob"]].to_numpy()

    return {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "results_through": conn.execute("SELECT MAX(match_date) FROM matches").fetchone()[0],
        "matches_in_database": conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
        "test": {
            "seasons": sorted(predictions["season"].unique().tolist()),
            "matches": len(predictions),
            "model": rounded(model.score(model_probs, predictions["result"])),
            "market": rounded(model.score(predictions[model.MARKET].to_numpy(), predictions["result"])),
        },
        "backtest": {
            **rounded(backtest.summarise(bets)),
            "luck_test_share_as_good": round(backtest.luck_test(bets), 3),
        },
        "upcoming_fixtures": len(upcoming),
    }


def main() -> None:
    for number, (name, step) in enumerate(STEPS, start=1):
        print(f"\n=== Step {number}/{len(STEPS)}: {name} ===")
        step()

    with sqlite3.connect(DB_PATH) as conn:
        result = summary(conn)
    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n=== Done. Summary saved to {SUMMARY_PATH.relative_to(ROOT)} ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
