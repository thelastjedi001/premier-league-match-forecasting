-- Phase 2: SQL exploration of Premier League results and bookmaker odds
-- Database: data/football.db (table: matches), built by src/ingest.py


-- Warm-up: Which team scored the most away goals?
SELECT away_team, SUM(away_goals) AS total_away_goals
FROM matches
GROUP BY away_team
ORDER BY total_away_goals DESC;


-- Average home shots on target per game (fair comparison across teams)
-- Totals favour teams with more games; averages fix that (promoted teams have far fewer).
SELECT home_team,
       COUNT(*) AS games,
       SUM(home_shots_on_target) AS total_shots_on_target,
       AVG(home_shots_on_target) AS avg_shots_on_target
FROM matches
GROUP BY home_team
ORDER BY avg_shots_on_target DESC;


-- Q1: Has home advantage changed over the seasons?
-- FINDING: Home win % ranges from 40% to 48% across seasons.
-- Peaked in 2022-23 (48.4%), lower since. Draws rising (~22% -> ~27%).
-- 2026-27 is only 90 matches in, so treat it with caution.
SELECT season,
       COUNT(*) AS matches,
       SUM(result = 'H') AS home_wins,
       SUM(result = 'D') AS draws,
       SUM(result = 'A') AS away_wins,
       ROUND(100.0 * SUM(result = 'H') / COUNT(*), 1) AS home_win_pct,
       ROUND(100.0 * SUM(result = 'D') / COUNT(*), 1) AS draw_pct,
       ROUND(100.0 * SUM(result = 'A') / COUNT(*), 1) AS away_win_pct
FROM matches
GROUP BY season
ORDER BY season;


-- Q2: Are bookmakers accurate? (calibration check)
-- FINDING: Bookmaker probabilities are well calibrated (predicted ~ actual in every band),
-- but predicted is usually a few points HIGHER than actual: evidence of the bookmaker margin.
-- Extreme bands (e.g. 90%+, only 19 matches) are too small to trust.
SELECT CAST(100.0 / avg_home_odds / 10 AS INTEGER) * 10 AS prob_band,
       COUNT(*) AS matches,
       ROUND(AVG(100.0 / avg_home_odds), 1) AS avg_predicted_pct,
       ROUND(100.0 * SUM(result = 'H') / COUNT(*), 1) AS actual_home_win_pct
FROM matches
GROUP BY prob_band
ORDER BY prob_band;


-- Q3: How big is the bookmaker's margin (overround)?
-- FINDING: Margin was flat at 4.1% for three seasons (2021-24),
-- then rose every season to 6.3% in 2026-27: a ~50% increase in the bookmaker's cut.
-- Bettors are getting worse value than ever. A model must beat this margin to profit.
SELECT season,
       COUNT(*) AS matches,
       ROUND(AVG(100.0 / avg_home_odds + 100.0 / avg_draw_odds + 100.0 / avg_away_odds), 1) AS avg_overround_pct
FROM matches
GROUP BY season
ORDER BY season;


-- Q4: Which teams beat the market's expectations? (home games, 2025-26)
-- Fair probability = home chance / total of all three chances (removes the margin).
-- FINDING: Tottenham won 3 home games vs 8.1 expected (-5.1): the market overrated them all season.
-- Fulham won 11 vs 7.3 expected (+3.7). Possible "reputation bias" (big clubs overrated).
-- Needs checking across seasons.
SELECT home_team,
       COUNT(*) AS matches,
       ROUND(SUM((1 / avg_home_odds) / (1 / avg_home_odds + 1 / avg_draw_odds + 1 / avg_away_odds)), 1) AS expected_home_wins,
       SUM(result = 'H') AS actual_home_wins,
       ROUND(SUM(result = 'H') - SUM((1 / avg_home_odds) / (1 / avg_home_odds + 1 / avg_draw_odds + 1 / avg_away_odds)), 1) AS difference
FROM matches
WHERE season = '2025-26'
GROUP BY home_team
ORDER BY difference DESC;


-- Q5: Biggest home-win upsets (longest odds that still won)
-- FINDING: Biggest upset: Nott'm Forest 1-0 Liverpool (2022-23), Forest given just 8.8%.
-- Liverpool were on the wrong end of 4 of the 10 biggest home upsets.
-- All top-10 upsets were against big clubs, consistent with the Q4 "reputation bias" idea.
SELECT season, match_date, home_team, away_team,
       home_goals, away_goals,
       avg_home_odds,
       ROUND(100.0 / avg_home_odds, 1) AS home_prob_pct
FROM matches
WHERE result = 'H'
ORDER BY avg_home_odds DESC
LIMIT 10;
