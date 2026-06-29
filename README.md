# FIFA World Cup 2026 Match Outcome Predictor

**Line 1 — Best Model:** Logistic Regression achieved the highest test accuracy of **54.7%** on the 2022 FIFA World Cup test set (64 matches), outperforming Random Forest (53.1%) and XGBoost (50.0%).

**Line 2 — Custom Features:** `experience_diff` (home team's prior World Cup appearances minus away team's) captures tournament pressure and experience advantage; `rank_diff` / `points_diff` (away_rank − home_rank and home_points − away_points) quantify relative FIFA ranking and form gaps between teams.

**Line 3 — Key EDA Insight:** Brazil leads all nations in historical World Cup goals scored with **237 total goals** across 1930–2022, followed by Argentina (152), France (136), Italy (128), and Germany (126). Home teams win ~55% of all matches, highlighting a significant home/neutral-host advantage.

**Line 4 — Clustering Finding:** K-Means (k=3) clustering by avg goals scored, conceded, win rate, and matches played identified the following **Elite Contenders** (highest win rate cluster): Argentina, Brazil, France, Germany, Italy, Netherlands, Spain, England, Uruguay, West Germany, Portugal, Croatia, Belgium, Hungary, Sweden, Poland, Yugoslavia, Soviet Union, Switzerland, Austria, Czechoslovakia, Mexico, Turkey.

**Line 5 — Run the App:** `streamlit run deployment/app.py`
