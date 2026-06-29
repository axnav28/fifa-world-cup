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
    page_title="FIFA World Cup 2026 Match Predictor",
    page_icon=None,
    layout="wide",
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
    ELITE_TEAMS        = meta.get('elite_contenders', [])
    TOP_COUNTRY        = meta.get('top_scoring_country', 'Brazil')
    TOP_GOALS          = meta.get('top_scoring_goals', 237)
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    st.error(f"Model or data files not found. Please run the full notebook first.\n\n`{e}`")
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

# ── Custom Premium Styling (FIFA Theme - Deep Blue & Gold) ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown {
    font-family: 'Inter', sans-serif;
}

/* Background layout gradient */
.stApp {
    background: radial-gradient(circle at 50% 0%, #0d162d 0%, #050814 100%);
    color: #f1f5f9;
}

/* Header UI Elements */
.header-container {
    text-align: center;
    padding: 3rem 1rem 1.5rem 1rem;
    background: linear-gradient(180deg, rgba(13, 22, 45, 0.8) 0%, rgba(5, 8, 20, 0) 100%);
    border-bottom: 1px solid rgba(212, 175, 55, 0.15);
    margin-bottom: 2.5rem;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
    color: #ffffff;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.subtitle {
    font-size: 1.1rem;
    color: #c5a059; /* Muted Gold */
    margin-top: 0.5rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.model-badge-container {
    display: inline-flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 1.5rem;
}

.badge {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    padding: 0.5rem 1.2rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.badge-highlight {
    background: rgba(212, 175, 55, 0.05);
    border: 1px solid rgba(212, 175, 55, 0.25);
    color: #d4af37;
}

/* Glassmorphism container for selectors */
.selector-card {
    background: rgba(13, 22, 45, 0.45);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(212, 175, 55, 0.1);
    border-radius: 12px;
    padding: 2.5rem;
    box-shadow: 0 15px 35px rgba(0,0,0,0.6);
    margin-bottom: 2.5rem;
}

/* Outcome Result Cards */
.result-box {
    background: rgba(13, 22, 45, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.result-box-winner {
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.08) 0%, rgba(13, 22, 45, 0.8) 100%);
    border: 1px solid rgba(212, 175, 55, 0.4);
    box-shadow: 0 8px 30px rgba(212, 175, 55, 0.1);
}

.result-prob {
    font-size: 3rem;
    font-weight: 800;
    margin: 0.6rem 0;
    line-height: 1;
}

.result-label {
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 2px;
    color: #94a3b8;
    font-weight: 700;
}

.result-team {
    font-size: 1rem;
    color: #ffffff;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Custom Interactive Progress Bar */
.custom-progress-container {
    margin: 2.5rem 0;
    background: rgba(13, 22, 45, 0.3);
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.03);
}

.progress-label-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.progress-track {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    overflow: hidden;
    display: flex;
}

.progress-fill-home {
    height: 100%;
    background: #0f4c81; /* FIFA Classic Blue */
    transition: width 0.8s ease;
}

.progress-fill-draw {
    height: 100%;
    background: #475569; /* Muted slate */
    transition: width 0.8s ease;
}

.progress-fill-away {
    height: 100%;
    background: #8b0000; /* Dark Red */
    transition: width 0.8s ease;
}

/* Team Comparison Metric Cards */
.comparison-card {
    background: rgba(13, 22, 45, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 6px;
    padding: 1.5rem;
    text-align: center;
}

.comparison-val {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}

.comparison-lbl {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* General Streamlit Overrides */
div[data-baseweb="select"] > div {
    background-color: rgba(13, 22, 45, 0.7) !important;
    border-color: rgba(212, 175, 55, 0.15) !important;
    color: #ffffff !important;
    border-radius: 4px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 3rem;
    background-color: transparent;
    justify-content: center;
    border-bottom: 1px solid rgba(212, 175, 55, 0.15);
    padding-bottom: 0.5rem;
    margin-bottom: 2rem;
}

.stTabs [data-baseweb="tab"] {
    font-size: 1rem;
    font-weight: 700;
    color: #94a3b8;
    background-color: transparent !important;
    border: none !important;
    padding: 0.6rem 1.5rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    transition: color 0.3s;
}

.stTabs [aria-selected="true"] {
    color: #d4af37 !important;
    border-bottom: 2px solid #d4af37 !important;
}

.verdict-box {
    background: rgba(212, 175, 55, 0.04);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 6px;
    padding: 1.2rem;
    margin-top: 2rem;
    color: #d4af37;
    font-weight: 600;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ── Header Section ────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="main-title">FIFA World Cup 2026</div>
    <div class="subtitle">Match Outcome & Probability Forecast Engine</div>
    <div class="model-badge-container">
        <span class="badge">Model: {best_model_name}</span>
        <span class="badge badge-highlight">Test Accuracy: {accuracy:.1f}% on 2022 WC</span>
    </div>
</div>
""".format(
    best_model_name=MODEL_NAME,
    accuracy=MODEL_ACC * 100
), unsafe_allow_html=True)

# ── Create Tabs ──────────────────────────────────────────────────────────────
tab_predict, tab_compare, tab_insights = st.tabs([
    "Match Predictor",
    "Head-to-Head Comparison",
    "Tournament Insights"
])

# 🚀 TAB 1: MATCH PREDICTOR
with tab_predict:
    st.markdown('<div class="selector-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Home Team")
        home_team = st.selectbox("Select Home Team", all_teams, index=0, key="home_select")
    with c2:
        st.markdown("### Away Team")
        away_team = st.selectbox("Select Away Team", all_teams, index=1, key="away_select")
    
    st.markdown('</div>', unsafe_allow_html=True)

    if home_team == away_team:
        st.warning("Please select two different teams to make a prediction.")
    else:
        # Prediction calculation
        X_pred = build_feature_vector(home_team, away_team)
        probs  = model.predict_proba(X_pred)[0]
        classes = model.classes_

        prob_map = {int(c): float(p) for c, p in zip(classes, probs)}
        home_win_pct = prob_map.get(0, 0) * 100
        away_win_pct = prob_map.get(1, 0) * 100
        draw_pct     = prob_map.get(2, 0) * 100

        max_pct = max(home_win_pct, away_win_pct, draw_pct)
        is_home_winner = (home_win_pct == max_pct)
        is_draw_winner = (draw_pct == max_pct)
        is_away_winner = (away_win_pct == max_pct)

        # Layout for outcome cards
        o1, o2, o3 = st.columns(3)
        
        with o1:
            winner_class = "result-box-winner" if is_home_winner else ""
            st.markdown(f"""
            <div class="result-box {winner_class}">
                <div class="result-label">Home Win</div>
                <div class="result-prob" style="color: #0f4c81;">{home_win_pct:.1f}%</div>
                <div class="result-team">{home_team}</div>
            </div>
            """, unsafe_allow_html=True)

        with o2:
            winner_class = "result-box-winner" if is_draw_winner else ""
            st.markdown(f"""
            <div class="result-box {winner_class}">
                <div class="result-label">Draw</div>
                <div class="result-prob" style="color: #94a3b8;">{draw_pct:.1f}%</div>
                <div class="result-team">Close Match</div>
            </div>
            """, unsafe_allow_html=True)

        with o3:
            winner_class = "result-box-winner" if is_away_winner else ""
            st.markdown(f"""
            <div class="result-box {winner_class}">
                <div class="result-label">Away Win</div>
                <div class="result-prob" style="color: #8b0000;">{away_win_pct:.1f}%</div>
                <div class="result-team">{away_team}</div>
            </div>
            """, unsafe_allow_html=True)

        # Dynamic visually appealing outcome distribution bar
        st.markdown(f"""
        <div class="custom-progress-container">
            <div class="progress-label-row">
                <span>{home_team} ({home_win_pct:.1f}%)</span>
                <span>Draw ({draw_pct:.1f}%)</span>
                <span>{away_team} ({away_win_pct:.1f}%)</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill-home" style="width: {home_win_pct}%"></div>
                <div class="progress-fill-draw" style="width: {draw_pct}%"></div>
                <div class="progress-fill-away" style="width: {away_win_pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show final verdict
        verdict = f"{home_team}" if is_home_winner else (f"{away_team}" if is_away_winner else "Draw")
        st.markdown(f"""
        <div class="verdict-box">
            Verdict: The prediction engine favors {verdict} with a {max_pct:.1f}% confidence level.
        </div>
        """, unsafe_allow_html=True)

# 📊 TAB 2: HEAD-TO-HEAD COMPARISON
with tab_compare:
    st.markdown("<h3 style='text-align:center;margin-bottom:2rem;text-transform:uppercase;letter-spacing:1px;font-size:1.3rem;'>Side-by-Side Analytics</h3>", unsafe_allow_html=True)

    h_rank, h_pts, h_wr, h_exp = get_team_features(home_team, MEDIAN_RANK, MEDIAN_POINTS, home_wr_dict, GLOBAL_HOME_WR_MED)
    a_rank, a_pts, a_wr, a_exp = get_team_features(away_team, MEDIAN_RANK, MEDIAN_POINTS, away_wr_dict, GLOBAL_AWAY_WR_MED)

    # Multi-column grid for head-to-head comparison
    col_h_stats, col_label, col_a_stats = st.columns([4, 2, 4])
    
    with col_h_stats:
        st.markdown(f"<h4 style='color:#0f4c81;text-align:right;text-transform:uppercase;letter-spacing:1px;'>{home_team}</h4>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="comparison-card" style="margin-bottom:1rem;">
            <div class="comparison-val">#{int(h_rank)}</div>
            <div class="comparison-lbl">FIFA World Rank</div>
        </div>
        <div class="comparison-card" style="margin-bottom:1rem;">
            <div class="comparison-val">{h_pts:.0f}</div>
            <div class="comparison-lbl">FIFA Points</div>
        </div>
        <div class="comparison-card" style="margin-bottom:1rem;">
            <div class="comparison-val">{h_wr:.1%}</div>
            <div class="comparison-lbl">Historical Home Win Rate</div>
        </div>
        <div class="comparison-card">
            <div class="comparison-val">{int(h_exp)}</div>
            <div class="comparison-lbl">World Cup Appearances</div>
        </div>
        """, unsafe_allow_html=True)

    with col_label:
        st.markdown("<p style='text-align:center;font-size:1.1rem;font-weight:700;margin-top:2.8rem;color:#64748b;'>VS</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;font-size:1.1rem;font-weight:700;margin-top:4.8rem;color:#64748b;'>VS</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;font-size:1.1rem;font-weight:700;margin-top:4.8rem;color:#64748b;'>VS</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;font-size:1.1rem;font-weight:700;margin-top:4.8rem;color:#64748b;'>VS</p>", unsafe_allow_html=True)

    with col_a_stats:
        st.markdown(f"<h4 style='color:#8b0000;text-align:left;text-transform:uppercase;letter-spacing:1px;'>{away_team}</h4>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="comparison-card" style="margin-bottom:1rem;">
            <div class="comparison-val">#{int(a_rank)}</div>
            <div class="comparison-lbl">FIFA World Rank</div>
        </div>
        <div class="comparison-card" style="margin-bottom:1rem;">
            <div class="comparison-val">{a_pts:.0f}</div>
            <div class="comparison-lbl">FIFA Points</div>
        </div>
        <div class="comparison-card" style="margin-bottom:1rem;">
            <div class="comparison-val">{a_wr:.1%}</div>
            <div class="comparison-lbl">Historical Away Win Rate</div>
        </div>
        <div class="comparison-card">
            <div class="comparison-val">{int(a_exp)}</div>
            <div class="comparison-lbl">World Cup Appearances</div>
        </div>
        """, unsafe_allow_html=True)

# 🏆 TAB 3: TOURNAMENT INSIGHTS
with tab_insights:
    st.markdown("### Historical Insights & Clusters")
    
    ins_col1, ins_col2 = st.columns(2)
    with ins_col1:
        st.markdown("""
        <div class="selector-card">
            <h4 style="text-transform:uppercase;color:#d4af37;letter-spacing:1px;margin-bottom:1rem;">Historical Top Goalscorers</h4>
            <p>Brazil leads all nations in historical World Cup goals scored with <strong>{goals} total goals</strong>.</p>
            <p>Home teams win <strong>~55%</strong> of all matches, showing a massive advantage.</p>
        </div>
        """.format(goals=TOP_GOALS), unsafe_allow_html=True)
        
        # Display the goalscorers visual if it exists
        img_goals_path = os.path.join(BASE_DIR, 'top10_goals.png')
        if os.path.exists(img_goals_path):
            st.image(img_goals_path, use_container_width=True)

    with ins_col2:
        st.markdown("""
        <div class="selector-card">
            <h4 style="text-transform:uppercase;color:#d4af37;letter-spacing:1px;margin-bottom:1rem;">K-Means Team Clusters (k=3)</h4>
            <p>Teams were clustered into <strong>Elite Contenders</strong>, <strong>Mid-Table</strong>, and <strong>Underdogs</strong>.</p>
            <p>Elite Contenders include: <code>{elite_list}</code></p>
        </div>
        """.format(elite_list=", ".join(ELITE_TEAMS[:10]) + "..."), unsafe_allow_html=True)

        img_cluster_path = os.path.join(BASE_DIR, 'team_clusters.png')
        if os.path.exists(img_cluster_path):
            st.image(img_cluster_path, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top: 1px solid rgba(212,175,55,0.15); margin-top: 5rem; padding: 2.5rem 0; text-align: center;">
    <p style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin:0;">
        FIFA World Cup 2026 Prediction Challenge · Model: Logistic Regression · Data: FIFA Rankings & Match History 1930–2022
    </p>
</div>
""", unsafe_allow_html=True)
