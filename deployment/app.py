"""
FIFA World Cup 2026 Match Predictor — Streamlit App
----------------------------------------------------
Run: streamlit run deployment/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA WC 2026 Predictor",
    page_icon="⚽",
    layout="centered",
)

# ── Resolve paths relative to repo root ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, 'deployment', 'model.pkl')
    return joblib.load(model_path)

@st.cache_data
def load_data():
    rank26   = pd.read_csv(os.path.join(BASE_DIR, 'data', 'rank26_clean.csv'))
    wr_df    = pd.read_csv(os.path.join(BASE_DIR, 'data', 'team_win_rates.csv'))
    exp_df   = pd.read_csv(os.path.join(BASE_DIR, 'data', 'team_experience.csv'))
    schedule = pd.read_csv(os.path.join(BASE_DIR, 'data', 'schedule_2026.csv'))
    with open(os.path.join(BASE_DIR, 'data', 'app_meta.json')) as f:
        meta = json.load(f)
    return rank26, wr_df, exp_df, schedule, meta

# ── Load resources ────────────────────────────────────────────────────────────
try:
    model = load_model()
    rank26, wr_df, exp_df, schedule, meta = load_data()
    FEATURE_COLS       = meta['feature_cols']
    MEDIAN_RANK        = meta['median_rank']
    MEDIAN_POINTS      = meta['median_points']
    GLOBAL_HOME_WR_MED = meta['global_home_wr_med']
    GLOBAL_AWAY_WR_MED = meta['global_away_wr_med']
    MODEL_NAME         = meta['best_model_name']
    MODEL_ACC          = meta['best_model_accuracy']
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    st.error(f"⚠️ Model or data files not found. Please run the full notebook first.\n\n`{e}`")
    st.stop()

# ── Extract all 2026 teams ────────────────────────────────────────────────────
all_teams = sorted(set(schedule['home_team'].tolist() + schedule['away_team'].tolist()))

# ── Build lookup dicts ────────────────────────────────────────────────────────
rank26_dict   = rank26.set_index('team')[['rank', 'points']].to_dict('index')
home_wr_dict  = wr_df.set_index('team')['home_win_rate'].to_dict()
away_wr_dict  = wr_df.set_index('team')['away_win_rate'].to_dict()
exp_dict      = exp_df.set_index('team')['experience'].to_dict()

def get_team_features(team, median_rank, median_points, win_rate_dict, global_wr_med):
    info = rank26_dict.get(team, {})
    rank   = info.get('rank',   median_rank)
    points = info.get('points', median_points)
    wr     = win_rate_dict.get(team, global_wr_med)
    exp    = exp_dict.get(team, 0)
    return rank, points, wr, exp

def build_feature_vector(home_team, away_team):
    h_rank, h_pts, h_wr, h_exp = get_team_features(
        home_team, MEDIAN_RANK, MEDIAN_POINTS, home_wr_dict, GLOBAL_HOME_WR_MED)
    a_rank, a_pts, a_wr, a_exp = get_team_features(
        away_team, MEDIAN_RANK, MEDIAN_POINTS, away_wr_dict, GLOBAL_AWAY_WR_MED)

    rank_diff      = a_rank - h_rank
    points_diff    = h_pts  - a_pts
    experience_diff = h_exp - a_exp

    row = pd.DataFrame([{
        'home_rank':          h_rank,
        'away_rank':          a_rank,
        'home_points':        h_pts,
        'away_points':        a_pts,
        'rank_diff':          rank_diff,
        'points_diff':        points_diff,
        'home_team_win_rate': h_wr,
        'away_team_win_rate': a_wr,
        'experience_diff':    experience_diff,
    }])
    return row[FEATURE_COLS]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
}

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
}

.hero-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f4d03f, #f39c12, #e74c3c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    text-align: center;
    color: #8b949e;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.model-badge {
    text-align: center;
    background: rgba(244, 208, 63, 0.1);
    border: 1px solid rgba(244, 208, 63, 0.3);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    color: #f4d03f;
    margin-bottom: 2rem;
}

.result-card {
    background: rgba(22, 27, 34, 0.9);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
}

.result-card h2 {
    font-size: 2.5rem;
    margin: 0;
    font-weight: 800;
}

.result-card p {
    color: #8b949e;
    margin: 0.2rem 0 0 0;
    font-size: 0.9rem;
}

.stSelectbox label {
    color: #c9d1d9 !important;
    font-weight: 600;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #f4d03f, #f39c12);
    color: #0d1117;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.75rem;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.85;
}

.divider {
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">⚽ FIFA World Cup 2026</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Match Outcome Predictor — Powered by Machine Learning</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="model-badge">🤖 Model: <strong>{MODEL_NAME}</strong> &nbsp;|&nbsp; '
    f'Test Accuracy: <strong>{MODEL_ACC*100:.1f}%</strong> (on 2022 WC)</div>',
    unsafe_allow_html=True
)

# ── Team selectors ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("🏠 **Home Team**")
    home_team = st.selectbox("Select Home Team", all_teams, index=0, key="home_sel",
                              label_visibility="collapsed")
with col2:
    st.markdown("✈️ **Away Team**")
    away_team = st.selectbox("Select Away Team", all_teams, index=1, key="away_sel",
                              label_visibility="collapsed")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Predict button ─────────────────────────────────────────────────────────────
if st.button("⚡ Predict Match Outcome", use_container_width=True):
    if home_team == away_team:
        st.warning("⚠️ Please select two different teams.")
    else:
        X_pred = build_feature_vector(home_team, away_team)
        probs  = model.predict_proba(X_pred)[0]
        classes = model.classes_

        prob_map = {int(c): float(p) for c, p in zip(classes, probs)}
        home_win_pct = prob_map.get(0, 0) * 100
        away_win_pct = prob_map.get(1, 0) * 100
        draw_pct     = prob_map.get(2, 0) * 100

        st.markdown("### 📊 Predicted Outcome Probabilities")
        c1, c2, c3 = st.columns(3)

        with c1:
            color = "#f4d03f" if home_win_pct == max(home_win_pct, away_win_pct, draw_pct) else "#c9d1d9"
            st.markdown(f"""
            <div class="result-card">
                <p>🏠 Home Win</p>
                <h2 style="color:{color}">{home_win_pct:.1f}%</h2>
                <p style="margin-top:0.3rem;font-size:0.8rem;">{home_team}</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            color = "#f4d03f" if draw_pct == max(home_win_pct, away_win_pct, draw_pct) else "#c9d1d9"
            st.markdown(f"""
            <div class="result-card">
                <p>🤝 Draw</p>
                <h2 style="color:{color}">{draw_pct:.1f}%</h2>
                <p style="margin-top:0.3rem;font-size:0.8rem;">Either team</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            color = "#f4d03f" if away_win_pct == max(home_win_pct, away_win_pct, draw_pct) else "#c9d1d9"
            st.markdown(f"""
            <div class="result-card">
                <p>✈️ Away Win</p>
                <h2 style="color:{color}">{away_win_pct:.1f}%</h2>
                <p style="margin-top:0.3rem;font-size:0.8rem;">{away_team}</p>
            </div>
            """, unsafe_allow_html=True)

        # Verdict
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        best_outcome = max(
            [("🏠 Home Win", home_win_pct, home_team),
             ("🤝 Draw",     draw_pct, ""),
             ("✈️ Away Win", away_win_pct, away_team)],
            key=lambda x: x[1]
        )
        team_str = f" — **{best_outcome[2]}**" if best_outcome[2] else ""
        st.success(f"**Most likely outcome:** {best_outcome[0]}{team_str} ({best_outcome[1]:.1f}%)")

        # Feature debug expander
        with st.expander("🔍 View Input Features"):
            h_rank   = rank26_dict.get(home_team, {}).get('rank', MEDIAN_RANK)
            a_rank   = rank26_dict.get(away_team, {}).get('rank', MEDIAN_RANK)
            h_pts    = rank26_dict.get(home_team, {}).get('points', MEDIAN_POINTS)
            a_pts    = rank26_dict.get(away_team, {}).get('points', MEDIAN_POINTS)
            st.markdown(f"""
| Feature | {home_team} (Home) | {away_team} (Away) |
|---------|-------------------|-------------------|
| FIFA Rank | {int(h_rank)} | {int(a_rank)} |
| FIFA Points | {h_pts:.0f} | {a_pts:.0f} |
| Win Rate | {home_wr_dict.get(home_team, GLOBAL_HOME_WR_MED):.2%} | {away_wr_dict.get(away_team, GLOBAL_AWAY_WR_MED):.2%} |
| WC Experience | {int(exp_dict.get(home_team, 0))} | {int(exp_dict.get(away_team, 0))} |
""")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#484f58;font-size:0.78rem;">'
    'Built with ❤️ · Trained on FIFA World Cup data 1930–2022 · '
    'Predictions are probabilistic estimates only.'
    '</p>',
    unsafe_allow_html=True
)
