"""
FIFA World Cup 2026 — Full ML Pipeline Script
Runs all steps, saves outputs, prints metrics for README.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import os
import joblib

warnings.filterwarnings('ignore')

DATA_DIR = 'data/'
print("="*60)
print("STEP 1: DATA INGESTION & COUNTRY NAME STANDARDIZATION")
print("="*60)

# ── Load all 5 CSV files ──────────────────────────────────────────────────────
matches   = pd.read_csv(DATA_DIR + 'matches_1930_2022.csv')
world_cup = pd.read_csv(DATA_DIR + 'world_cup.csv')
rank22    = pd.read_csv(DATA_DIR + 'fifa_ranking_2022-10-06.csv')
rank26    = pd.read_csv(DATA_DIR + 'fifa_ranking_2026-06-08.csv')
schedule  = pd.read_csv(DATA_DIR + 'schedule_2026.csv')

print(f"matches:   {matches.shape}")
print(f"world_cup: {world_cup.shape}")
print(f"rank22:    {rank22.shape}")
print(f"rank26:    {rank26.shape}")
print(f"schedule:  {schedule.shape}")

# ── Country name mapping ──────────────────────────────────────────────────────
name_map = {
    'USA':                    'United States',
    'United States of America': 'United States',
    'Korea Republic':         'South Korea',
    'Korea DPR':              'North Korea',
    'IR Iran':                'Iran',
    'Türkiye':                'Turkey',
    'Czech Republic':         'Czechia',
    'Trinidad and Tobago':    'Trinidad & Tobago',
    'Cabo Verde':             'Cape Verde',
    'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'German DR':              'East Germany',
    'China PR':               'China',
    "Côte d'Ivoire":          'Ivory Coast',
    "Cote d'Ivoire":          'Ivory Coast',
    'Curacao':                'Curaçao',
}

def apply_name_map(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(name_map)
    return df

matches  = apply_name_map(matches,  ['home_team', 'away_team'])
rank22   = apply_name_map(rank22,   ['team'])
rank26   = apply_name_map(rank26,   ['team'])
schedule = apply_name_map(schedule, ['home_team', 'away_team'])

# Verify year column already exists
print(f"Unique Years: {sorted(matches['Year'].unique())}")

# Assert zero NaN in critical columns
for col in ['home_team', 'away_team', 'home_score', 'away_score']:
    assert matches[col].isnull().sum() == 0, f"NaN in {col}!"
print("✅ Step 1: Zero NaN assertion passed.")

print("\n" + "="*60)
print("STEP 2: MERGING FIFA RANKINGS")
print("="*60)

rank22_lookup = rank22[['team', 'rank', 'points']].copy()

matches = matches.merge(
    rank22_lookup.rename(columns={'team': 'home_team', 'rank': 'home_rank', 'points': 'home_points'}),
    on='home_team', how='left'
)
matches = matches.merge(
    rank22_lookup.rename(columns={'team': 'away_team', 'rank': 'away_rank', 'points': 'away_points'}),
    on='away_team', how='left'
)

median_rank   = rank22_lookup['rank'].median()
median_points = rank22_lookup['points'].median()
print(f"Median rank={median_rank}, median_points={median_points:.2f}")

matches['home_rank']   = matches['home_rank'].fillna(median_rank)
matches['away_rank']   = matches['away_rank'].fillna(median_rank)
matches['home_points'] = matches['home_points'].fillna(median_points)
matches['away_points'] = matches['away_points'].fillna(median_points)

for col in ['home_rank', 'away_rank', 'home_points', 'away_points']:
    assert matches[col].isnull().sum() == 0
print("✅ Step 2: Ranking merge assertions passed.")

print("\n" + "="*60)
print("STEP 3: TARGET VARIABLE & EDA")
print("="*60)

label_map = {0: 'Home Win', 1: 'Away Win', 2: 'Draw'}

def get_match_result(row):
    if row['home_score'] > row['away_score']: return 0
    elif row['away_score'] > row['home_score']: return 1
    else: return 2

matches['Match_Result'] = matches.apply(get_match_result, axis=1)

vc = matches['Match_Result'].value_counts().sort_index()
pct = (vc / len(matches) * 100).round(1)
print("Match Result Distribution:")
for k, v in vc.items():
    print(f"  {label_map[k]:10s}: {v:4d}  ({pct[k]}%)")

# EDA plots
stat_cols = ['home_score', 'away_score', 'home_rank', 'away_rank', 'home_points', 'away_points']
summary = matches[stat_cols].agg(['mean', 'median', 'std']).T.round(3)
print("\nSummary Statistics:")
print(summary)

# Correlation heatmap
corr_cols = ['home_score', 'away_score', 'home_rank', 'away_rank',
             'home_points', 'away_points', 'Match_Result']
corr = matches[corr_cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Correlation Matrix — FIFA World Cup Match Features', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

# Top 10 goals
home_goals = matches.groupby('home_team')['home_score'].sum().rename('goals')
away_goals = matches.groupby('away_team')['away_score'].sum().rename('goals')
total_goals = (home_goals.add(away_goals, fill_value=0)
               .sort_values(ascending=False).head(10))
print(f"\nTop 10 Countries by Goals:")
print(total_goals.to_string())

# Store the EDA insight
top_country = total_goals.index[0]
top_goals   = int(total_goals.iloc[0])

colors = sns.color_palette('viridis', 10)
plt.figure(figsize=(10, 6))
bars = plt.bar(total_goals.index, total_goals.values, color=colors, edgecolor='white', linewidth=0.8)
plt.title('Top 10 Countries by Total Historical Goals Scored\n(FIFA World Cup 1930–2022)',
          fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Total Goals', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
for bar, val in zip(bars, total_goals.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{int(val)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('top10_goals.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: top10_goals.png")

print("\n" + "="*60)
print("STEP 4: FEATURE ENGINEERING")
print("="*60)

matches = matches.sort_values('Year').reset_index(drop=True)
matches['rank_diff']   = matches['away_rank'] - matches['home_rank']
matches['points_diff'] = matches['home_points'] - matches['away_points']

matches['home_win_flag'] = (matches['Match_Result'] == 0).astype(int)
matches['away_win_flag'] = (matches['Match_Result'] == 1).astype(int)

matches['home_team_win_rate'] = (
    matches.groupby('home_team')['home_win_flag']
    .transform(lambda x: x.shift(1).expanding().mean())
)
matches['away_team_win_rate'] = (
    matches.groupby('away_team')['away_win_flag']
    .transform(lambda x: x.shift(1).expanding().mean())
)

# Tournament experience feature
home_appearances = matches[['home_team', 'Year']].rename(columns={'home_team': 'team'})
away_appearances = matches[['away_team', 'Year']].rename(columns={'away_team': 'team'})
all_appearances  = pd.concat([home_appearances, away_appearances]).drop_duplicates()

team_year_count = {}
for team, grp in all_appearances.groupby('team'):
    years_sorted = sorted(grp['Year'].unique())
    for i, yr in enumerate(years_sorted):
        team_year_count[(team, yr)] = i

matches['home_experience'] = matches.apply(
    lambda r: team_year_count.get((r['home_team'], r['Year']), 0), axis=1)
matches['away_experience'] = matches.apply(
    lambda r: team_year_count.get((r['away_team'], r['Year']), 0), axis=1)
matches['experience_diff'] = matches['home_experience'] - matches['away_experience']

all_feat_cols = ['home_rank', 'away_rank', 'home_points', 'away_points',
                 'rank_diff', 'points_diff', 'home_team_win_rate',
                 'away_team_win_rate', 'experience_diff']

for col in all_feat_cols:
    med = matches[col].median()
    matches[col] = matches[col].fillna(med)

print("✅ Step 4: Feature engineering complete. Zero NaN remaining.")

print("\n" + "="*60)
print("STEP 5: TEMPORAL TRAIN/TEST SPLIT")
print("="*60)

FEATURE_COLS = [
    'home_rank', 'away_rank', 'home_points', 'away_points',
    'rank_diff', 'points_diff',
    'home_team_win_rate', 'away_team_win_rate',
    'experience_diff'
]
TARGET_COL = 'Match_Result'

train = matches[matches['Year'] <= 2018].copy()
test  = matches[matches['Year'] == 2022].copy()

X_train = train[FEATURE_COLS]
y_train = train[TARGET_COL]
X_test  = test[FEATURE_COLS]
y_test  = test[TARGET_COL]

print(f"Train size: {len(train)} | Test size: {len(test)}")

print("\n" + "="*60)
print("STEP 6: CLASSIFICATION MODELS")
print("="*60)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def evaluate_classifier(model, X_tr, y_tr, X_te, y_te, name='Model'):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    metrics = {
        'Model':     name,
        'Accuracy':  accuracy_score(y_te, preds),
        'Precision': precision_score(y_te, preds, average='macro', zero_division=0),
        'Recall':    recall_score(y_te, preds, average='macro', zero_division=0),
        'F1-Score':  f1_score(y_te, preds, average='macro', zero_division=0),
    }
    return metrics, model, preds

results = []

# 6a. Logistic Regression (multi_class removed in sklearn>=1.5, now default is OvR/auto)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr_metrics, lr_model, lr_preds = evaluate_classifier(lr, X_train, y_train, X_test, y_test, 'Logistic Regression')
results.append(lr_metrics)
print(f"LR  Accuracy: {lr_metrics['Accuracy']:.4f}")

# Confusion matrix
plt.figure(figsize=(6, 5))
cm_lr = confusion_matrix(y_test, lr_preds)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(label_map.values()),
            yticklabels=list(label_map.values()))
plt.title('Logistic Regression — Confusion Matrix', fontsize=13)
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('cm_lr.png', dpi=120, bbox_inches='tight')
plt.close()

# 6b. Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_metrics, rf_model, rf_preds = evaluate_classifier(rf, X_train, y_train, X_test, y_test, 'Random Forest')
results.append(rf_metrics)
print(f"RF  Accuracy: {rf_metrics['Accuracy']:.4f}")

plt.figure(figsize=(6, 5))
cm_rf = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=list(label_map.values()),
            yticklabels=list(label_map.values()))
plt.title('Random Forest — Confusion Matrix', fontsize=13)
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('cm_rf.png', dpi=120, bbox_inches='tight')
plt.close()

# 6c. XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_metrics, xgb_model, xgb_preds = evaluate_classifier(xgb, X_train, y_train, X_test, y_test, 'XGBoost')
results.append(xgb_metrics)
print(f"XGB Accuracy: {xgb_metrics['Accuracy']:.4f}")

plt.figure(figsize=(6, 5))
cm_xgb = confusion_matrix(y_test, xgb_preds)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges',
            xticklabels=list(label_map.values()),
            yticklabels=list(label_map.values()))
plt.title('XGBoost — Confusion Matrix', fontsize=13)
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('cm_xgb.png', dpi=120, bbox_inches='tight')
plt.close()

# Model comparison table
results_df = pd.DataFrame(results).set_index('Model').round(4)
print("\nModel Comparison:")
print(results_df.to_string())

# 6d. GridSearchCV
print("\nRunning GridSearchCV...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                  cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
best_rf = gs.best_estimator_
best_rf_preds = best_rf.predict(X_test)
best_rf_acc = accuracy_score(y_test, best_rf_preds)
print(f"Best Params: {gs.best_params_}")
print(f"RF Accuracy before tuning: {rf_metrics['Accuracy']:.4f}")
print(f"RF Accuracy after  tuning: {best_rf_acc:.4f}")

# Determine best model
all_results_list = [
    ('Logistic Regression', lr_model,  lr_metrics['Accuracy']),
    ('Random Forest',       rf_model,  rf_metrics['Accuracy']),
    ('XGBoost',             xgb_model, xgb_metrics['Accuracy']),
    ('Tuned Random Forest', best_rf,   best_rf_acc),
]
best_name, best_model, best_acc = max(all_results_list, key=lambda x: x[2])
print(f"\nBest model: {best_name}  (accuracy={best_acc:.4f})")

os.makedirs('deployment', exist_ok=True)
joblib.dump(best_model, 'deployment/model.pkl')
print("✅ Saved: deployment/model.pkl")

# 6e. 2026 Probability Forecasting
print("\n--- 6e: 2026 Probability Forecasting ---")
fixtures_2026 = schedule[schedule['Round'] == 'Group stage'].head(10).copy()
rank26_lookup = rank26[['team', 'rank', 'points']].copy()

fixtures_2026 = fixtures_2026.merge(
    rank26_lookup.rename(columns={'team': 'home_team', 'rank': 'home_rank', 'points': 'home_points'}),
    on='home_team', how='left'
)
fixtures_2026 = fixtures_2026.merge(
    rank26_lookup.rename(columns={'team': 'away_team', 'rank': 'away_rank', 'points': 'away_points'}),
    on='away_team', how='left'
)

med_rank26   = rank26_lookup['rank'].median()
med_points26 = rank26_lookup['points'].median()
fixtures_2026['home_rank']   = fixtures_2026['home_rank'].fillna(med_rank26)
fixtures_2026['away_rank']   = fixtures_2026['away_rank'].fillna(med_rank26)
fixtures_2026['home_points'] = fixtures_2026['home_points'].fillna(med_points26)
fixtures_2026['away_points'] = fixtures_2026['away_points'].fillna(med_points26)

fixtures_2026['rank_diff']   = fixtures_2026['away_rank'] - fixtures_2026['home_rank']
fixtures_2026['points_diff'] = fixtures_2026['home_points'] - fixtures_2026['away_points']

final_home_wr = matches.groupby('home_team')['home_team_win_rate'].last().to_dict()
final_away_wr = matches.groupby('away_team')['away_team_win_rate'].last().to_dict()
global_home_wr_med = matches['home_team_win_rate'].median()
global_away_wr_med = matches['away_team_win_rate'].median()

fixtures_2026['home_team_win_rate'] = fixtures_2026['home_team'].map(final_home_wr).fillna(global_home_wr_med)
fixtures_2026['away_team_win_rate'] = fixtures_2026['away_team'].map(final_away_wr).fillna(global_away_wr_med)

team_max_exp = {}
for (team, yr), cnt in team_year_count.items():
    if team not in team_max_exp or cnt > team_max_exp[team]:
        team_max_exp[team] = cnt

fixtures_2026['home_experience'] = fixtures_2026['home_team'].map(team_max_exp).fillna(0)
fixtures_2026['away_experience'] = fixtures_2026['away_team'].map(team_max_exp).fillna(0)
fixtures_2026['experience_diff'] = fixtures_2026['home_experience'] - fixtures_2026['away_experience']

X_2026 = fixtures_2026[FEATURE_COLS]
probs  = best_model.predict_proba(X_2026)
classes = best_model.classes_
home_win_idx = list(classes).index(0)
away_win_idx = list(classes).index(1)
draw_idx     = list(classes).index(2)

forecast_df = pd.DataFrame({
    'Home Team':  fixtures_2026['home_team'].values,
    'Away Team':  fixtures_2026['away_team'].values,
    'Home Win %': (probs[:, home_win_idx] * 100).round(1),
    'Away Win %': (probs[:, away_win_idx] * 100).round(1),
    'Draw %':     (probs[:, draw_idx]     * 100).round(1),
})
print(forecast_df.to_string(index=False))
forecast_df.to_csv('data/forecast_2026.csv', index=False)

print("\n" + "="*60)
print("STEP 7: REGRESSION — TOTAL GOALS")
print("="*60)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_train_reg = (train['home_score'] + train['away_score']).values
y_test_reg  = (test['home_score']  + test['away_score']).values

rfr = RandomForestRegressor(n_estimators=200, random_state=42)
rfr.fit(X_train, y_train_reg)
y_pred_reg = rfr.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
mae  = float(mean_absolute_error(y_test_reg, y_pred_reg))
print(f"RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6, edgecolors='white', linewidth=0.5, s=70, color='steelblue')
min_val = min(y_test_reg.min(), y_pred_reg.min()) - 0.5
max_val = max(y_test_reg.max(), y_pred_reg.max()) + 0.5
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Total Goals'); plt.ylabel('Predicted Total Goals')
plt.title(f'Actual vs Predicted Total Goals (RMSE={rmse:.3f}, MAE={mae:.3f})', fontsize=13)
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('regression_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: regression_scatter.png")

print("\n" + "="*60)
print("STEP 8: K-MEANS CLUSTERING")
print("="*60)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

home_stats = matches.groupby('home_team').agg(
    home_goals_scored  =('home_score', 'mean'),
    home_goals_conceded=('away_score', 'mean'),
    home_wins          =('home_win_flag', 'sum'),
    home_matches       =('home_team', 'count')
).reset_index().rename(columns={'home_team': 'team'})

away_stats = matches.groupby('away_team').agg(
    away_goals_scored  =('away_score', 'mean'),
    away_goals_conceded=('home_score', 'mean'),
    away_wins          =('away_win_flag', 'sum'),
    away_matches       =('away_team', 'count')
).reset_index().rename(columns={'away_team': 'team'})

team_stats = home_stats.merge(away_stats, on='team', how='outer').fillna(0)
team_stats['avg_goals_scored']   = (team_stats['home_goals_scored'] + team_stats['away_goals_scored']) / 2
team_stats['avg_goals_conceded'] = (team_stats['home_goals_conceded'] + team_stats['away_goals_conceded']) / 2
team_stats['total_wins']         = team_stats['home_wins'] + team_stats['away_wins']
team_stats['total_matches']      = team_stats['home_matches'] + team_stats['away_matches']
team_stats['win_rate']           = team_stats['total_wins'] / team_stats['total_matches']

cluster_feats = ['avg_goals_scored', 'avg_goals_conceded', 'win_rate', 'total_matches']
team_cluster_df = team_stats[['team'] + cluster_feats].dropna().copy()

scaler    = StandardScaler()
X_cluster = scaler.fit_transform(team_cluster_df[cluster_feats])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
team_cluster_df['cluster'] = kmeans.fit_predict(X_cluster)

cluster_win_rates    = team_cluster_df.groupby('cluster')['win_rate'].mean().sort_values(ascending=False)
cluster_labels_ordered = cluster_win_rates.index.tolist()
cluster_label_map = {
    cluster_labels_ordered[0]: 'Elite Contenders',
    cluster_labels_ordered[1]: 'Mid-Table',
    cluster_labels_ordered[2]: 'Underdogs',
}
team_cluster_df['cluster_name'] = team_cluster_df['cluster'].map(cluster_label_map)

elite_teams = sorted(team_cluster_df[team_cluster_df['cluster_name'] == 'Elite Contenders']['team'].tolist())
print(f"Elite Contenders: {elite_teams}")

color_map = {'Elite Contenders': '#e63946', 'Mid-Table': '#2a9d8f', 'Underdogs': '#457b9d'}
top10_by_matches = team_cluster_df.nlargest(10, 'total_matches')['team'].tolist()

plt.figure(figsize=(11, 8))
for label, grp in team_cluster_df.groupby('cluster_name'):
    plt.scatter(grp['avg_goals_scored'], grp['win_rate'],
                label=label, color=color_map[label], s=80, alpha=0.7, edgecolors='white', linewidth=0.5)

for _, row in team_cluster_df[team_cluster_df['team'].isin(top10_by_matches)].iterrows():
    plt.annotate(row['team'],
                 xy=(row['avg_goals_scored'], row['win_rate']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

plt.xlabel('Avg Goals Scored per Match', fontsize=12)
plt.ylabel('Win Rate', fontsize=12)
plt.title('FIFA World Cup — K-Means Team Clustering (k=3)\n1930–2022', fontsize=14, fontweight='bold')
plt.legend(title='Cluster', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('team_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: team_clusters.png")

# ── Save supporting data for Streamlit app ────────────────────────────────────
print("\n" + "="*60)
print("STEP 9: SAVING STREAMLIT APP DATA")
print("="*60)

rank26.to_csv('data/rank26_clean.csv', index=False)

win_rate_df = pd.DataFrame({
    'team': list(final_home_wr.keys()),
    'home_win_rate': list(final_home_wr.values())
})
away_wr_df = pd.DataFrame({
    'team': list(final_away_wr.keys()),
    'away_win_rate': list(final_away_wr.values())
})
wr_df = win_rate_df.merge(away_wr_df, on='team', how='outer')
wr_df.to_csv('data/team_win_rates.csv', index=False)

exp_records = [{'team': t, 'experience': v} for (t, yr), v in team_year_count.items()]
exp_df = pd.DataFrame(exp_records).groupby('team')['experience'].max().reset_index()
exp_df.to_csv('data/team_experience.csv', index=False)

meta = {
    'feature_cols': FEATURE_COLS,
    'median_rank':   float(med_rank26),
    'median_points': float(med_points26),
    'global_home_wr_med': float(global_home_wr_med),
    'global_away_wr_med': float(global_away_wr_med),
    'best_model_name': best_name,
    'best_model_accuracy': float(best_acc),
    'elite_contenders': elite_teams,
    'top_scoring_country': top_country,
    'top_scoring_goals': top_goals,
    'regression_rmse': rmse,
    'regression_mae':  mae,
}
with open('data/app_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("✅ All supporting files saved.")
print("\n" + "="*60)
print("PIPELINE COMPLETE — SUMMARY")
print("="*60)
print(f"Best model:          {best_name}")
print(f"Test accuracy:       {best_acc:.4f} ({best_acc*100:.1f}%)")
print(f"Top scoring country: {top_country} ({top_goals} goals)")
print(f"Elite Contenders:    {elite_teams[:5]}...")
print(f"Regression RMSE:     {rmse:.4f}")
print(f"Regression MAE:      {mae:.4f}")
