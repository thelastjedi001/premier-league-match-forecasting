# ⚽ Premier League Match Predictor & Odds Tracker

An end-to-end data project: an automated pipeline collects Premier League results and
bookmaker odds, a machine learning model predicts match outcomes, and a live dashboard
compares the model against the bookmakers to find "value bets".

> Status: **Phase 1 in progress** (data pipeline)

## Why this project
Bookmakers price football matches the way traders price stocks. This project combines
**football analytics**, **probability and risk thinking (finance)**, and a **fan-facing
product (entertainment)**.

## Project roadmap

| Phase | What | Skills shown | Status |
|---|---|---|---|
| 1. Data pipeline | Download 6 seasons of EPL results + odds into a SQL database | Python, SQL, data cleaning | 🟡 |
| 2. Exploration | SQL + pandas analysis: home advantage, bookmaker accuracy, team form | SQL, EDA, visualization | ⬜ |
| 3. Features | Rolling form, goals, shots, xG, Elo ratings, rest days | Feature engineering | ⬜ |
| 4. Model | Predict Home/Draw/Away probabilities (logistic regression → XGBoost) | ML, probability calibration | ⬜ |
| 5. Betting backtest | Compare model vs. bookmaker odds, simulate a betting strategy, track ROI | Finance thinking, statistics | ⬜ |
| 6. Automation | GitHub Actions runs the pipeline + predictions every day | MLOps, scheduling | ⬜ |
| 7. Dashboard | Streamlit app: upcoming fixtures, predictions, model track record | Deployment, storytelling | ⬜ |

## Data source
[football-data.co.uk](https://www.football-data.co.uk/): free match results, stats
(shots, corners, cards, xG) and odds from major bookmakers, updated twice a week.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/ingest.py
```

This creates `data/football.db`, a SQLite database. Open it in **DBeaver**:
*New Database Connection → SQLite → Path: `data/football.db`*.

## Project structure
```
football-predictor/
├── data/            # SQLite database (generated)
├── src/
│   └── ingest.py    # Phase 1: download + load match data
├── requirements.txt
└── README.md
```
