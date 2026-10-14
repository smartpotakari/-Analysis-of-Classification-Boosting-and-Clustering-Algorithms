"""
=============================================================================
MASTER MACHINE LEARNING PIPELINE: 12 ALGORITHMS
=============================================================================
This master script executes and validates all 12 algorithms end-to-end:
  - Part 1: Classification (4 Algorithms) on social_networks.csv
  - Part 2: Boosting (4 Algorithms) on insurance.csv
  - Part 3: Clustering (4 Algorithms) on mall_customers.csv
=============================================================================
"""

import os
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score
)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 85)
print(" MACHINE LEARNING 12-ALGORITHM BENCHMARK EXECUTION PIPELINE")
print("=" * 85)
print("Environment Check:")
print(f"  Python Version: {sys.version.split()[0]}")
print(f"  NumPy Version:  {np.__version__}")
print(f"  Pandas Version: {pd.__version__}")
import sklearn
print(f"  Scikit-Learn:   {sklearn.__version__}")
import xgboost
print(f"  XGBoost:        {xgboost.__version__}")
print("=" * 85)

# ---------------------------------------------------------------------------
# 1. CLASSIFICATION PIPELINE (SOCIAL NETWORK ADS)
# ---------------------------------------------------------------------------
print("\n" + "#" * 85)
print("# SECTION 1: CLASSIFICATION (4 ALGORITHMS) — SOCIAL NETWORK ADS")
print("#" * 85)

df_social = pd.read_csv('social_networks.csv')
print(f"Loaded 'social_networks.csv' with {df_social.shape[0]} rows, {df_social.shape[1]} columns.")

# Preprocessing
le_gender = LabelEncoder()
df_social['Gender_Encoded'] = le_gender.fit_transform(df_social['Gender'])
X_soc = df_social[['Age', 'EstimatedSalary', 'Gender_Encoded']].copy()
y_soc = df_social['Purchased'].copy()

X_soc_train, X_soc_test, y_soc_train, y_soc_test = train_test_split(
    X_soc, y_soc, test_size=0.20, random_state=42, stratify=y_soc
)

scaler_soc = StandardScaler()
X_soc_train_scaled = scaler_soc.fit_transform(X_soc_train)
X_soc_test_scaled = scaler_soc.transform(X_soc_test)

clf_models = {
    'Decision Tree': (DecisionTreeClassifier(criterion='entropy', random_state=42), False),
    'Random Forest': (RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42), False),
    'K-Nearest Neighbors (KNN)': (KNeighborsClassifier(n_neighbors=5), True),
    'Support Vector Machine (SVM)': (SVC(kernel='rbf', C=1.0, random_state=42), True)
}

classification_records = []
for name, (model, is_scaled) in clf_models.items():
    X_tr = X_soc_train_scaled if is_scaled else X_soc_train
    X_te = X_soc_test_scaled if is_scaled else X_soc_test
    
    model.fit(X_tr, y_soc_train)
    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)
    
    tr_acc = accuracy_score(y_soc_train, y_tr_pred)
    te_acc = accuracy_score(y_soc_test, y_te_pred)
    prec = precision_score(y_soc_test, y_te_pred, zero_division=0)
    rec = recall_score(y_soc_test, y_te_pred, zero_division=0)
    f1 = f1_score(y_soc_test, y_te_pred, zero_division=0)
    
    joblib.dump(model, f"models/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib")
    classification_records.append({
        'Category': 'Classification',
        'Algorithm': name,
        'Dataset': 'Social Network Ads',
        'Train Accuracy (%)': round(tr_acc * 100, 2),
        'Test Accuracy (%)': round(te_acc * 100, 2),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4)
    })

df_clf_summary = pd.DataFrame(classification_records)
print(df_clf_summary[['Algorithm', 'Train Accuracy (%)', 'Test Accuracy (%)', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))

# ---------------------------------------------------------------------------
# 2. BOOSTING PIPELINE (INSURANCE)
# ---------------------------------------------------------------------------
print("\n" + "#" * 85)
print("# SECTION 2: BOOSTING (4 ALGORITHMS) — INSURANCE DATASET")
print("#" * 85)

df_ins = pd.read_csv('insurance.csv')
print(f"Loaded 'insurance.csv' with {df_ins.shape[0]} rows, {df_ins.shape[1]} columns.")

# Target is 'smoker'
le_smoker = LabelEncoder()
df_ins['smoker_encoded'] = le_smoker.fit_transform(df_ins['smoker'])
le_sex = LabelEncoder()
df_ins['sex_encoded'] = le_sex.fit_transform(df_ins['sex'])

feature_cols_ins = ['age', 'sex_encoded', 'bmi', 'children', 'charges']
X_ins = df_ins[feature_cols_ins].copy()
y_ins = df_ins['smoker_encoded'].copy()

X_ins_train, X_ins_test, y_ins_train, y_ins_test = train_test_split(
    X_ins, y_ins, test_size=0.20, random_state=42, stratify=y_ins
)

boosting_models = {
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss'),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_depth=6, random_state=42)
}

boosting_records = []
for name, model in boosting_models.items():
    model.fit(X_ins_train, y_ins_train)
    y_tr_pred = model.predict(X_ins_train)
    y_te_pred = model.predict(X_ins_test)
    
    tr_acc = accuracy_score(y_ins_train, y_tr_pred)
    te_acc = accuracy_score(y_ins_test, y_te_pred)
    prec = precision_score(y_ins_test, y_te_pred, zero_division=0)
    rec = recall_score(y_ins_test, y_te_pred, zero_division=0)
    f1 = f1_score(y_ins_test, y_te_pred, zero_division=0)
    
    joblib.dump(model, f"models/{name.lower().replace(' ', '_')}.joblib")
    boosting_records.append({
        'Category': 'Boosting',
        'Algorithm': name,
        'Dataset': 'Insurance',
        'Train Accuracy (%)': round(tr_acc * 100, 2),
        'Test Accuracy (%)': round(te_acc * 100, 2),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4)
    })

df_boost_summary = pd.DataFrame(boosting_records)
print(df_boost_summary[['Algorithm', 'Train Accuracy (%)', 'Test Accuracy (%)', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))

# ---------------------------------------------------------------------------
# 3. CLUSTERING PIPELINE (MALL CUSTOMERS)
# ---------------------------------------------------------------------------
print("\n" + "#" * 85)
print("# SECTION 3: CLUSTERING (4 ALGORITHMS) — MALL CUSTOMERS")
print("#" * 85)

df_mall = pd.read_csv('mall_customers.csv')
print(f"Loaded 'mall_customers.csv' with {df_mall.shape[0]} rows, {df_mall.shape[1]} columns.")

X_mall = df_mall[['Annual Income (k$)', 'Spending Score (1-100)']].copy()
scaler_mall = StandardScaler()
X_mall_scaled = scaler_mall.fit_transform(X_mall)

k_opt = 5
km_clust = KMeans(n_clusters=k_opt, init='k-means++', random_state=42, n_init=10)
lbl_km = km_clust.fit_predict(X_mall_scaled)

agg_clust = AgglomerativeClustering(n_clusters=k_opt, linkage='ward')
lbl_agg = agg_clust.fit_predict(X_mall_scaled)

db_clust = DBSCAN(eps=0.35, min_samples=5)
lbl_db = db_clust.fit_predict(X_mall_scaled)

gmm_clust = GaussianMixture(n_components=k_opt, covariance_type='full', random_state=42)
lbl_gmm = gmm_clust.fit_predict(X_mall_scaled)

clustering_dict = {
    'K-Means': lbl_km,
    'Agglomerative Hierarchical': lbl_agg,
    'DBSCAN': lbl_db,
    'Gaussian Mixture Model (GMM)': lbl_gmm
}

clustering_records = []
for name, labels in clustering_dict.items():
    uniq = set(labels)
    n_cl = len(uniq - {-1})
    n_noise = list(labels).count(-1)
    
    if n_cl > 1:
        if -1 in uniq:
            mask = (labels != -1)
            sil = silhouette_score(X_mall_scaled[mask], labels[mask])
            db_idx = davies_bouldin_score(X_mall_scaled[mask], labels[mask])
            ch_idx = calinski_harabasz_score(X_mall_scaled[mask], labels[mask])
        else:
            sil = silhouette_score(X_mall_scaled, labels)
            db_idx = davies_bouldin_score(X_mall_scaled, labels)
            ch_idx = calinski_harabasz_score(X_mall_scaled, labels)
    else:
        sil, db_idx, ch_idx = np.nan, np.nan, np.nan
        
    clustering_records.append({
        'Category': 'Clustering',
        'Algorithm': name,
        'Dataset': 'Mall Customers',
        'Clusters Found': n_cl,
        'Noise Points': n_noise,
        'Silhouette Score': round(sil, 4) if not np.isnan(sil) else 'N/A',
        'Davies-Bouldin Index': round(db_idx, 4) if not np.isnan(db_idx) else 'N/A',
        'Calinski-Harabasz Index': round(ch_idx, 2) if not np.isnan(ch_idx) else 'N/A'
    })

df_clust_summary = pd.DataFrame(clustering_records)
print(df_clust_summary[['Algorithm', 'Clusters Found', 'Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']].to_string(index=False))

# ---------------------------------------------------------------------------
# 4. SAVE COMPREHENSIVE COMBINED RESULTS
# ---------------------------------------------------------------------------
df_clf_summary.to_csv('results/classification_results.csv', index=False)
df_boost_summary.to_csv('results/boosting_results.csv', index=False)
df_clust_summary.to_csv('results/clustering_results.csv', index=False)

# Combine supervised results
df_supervised = pd.concat([df_clf_summary, df_boost_summary], ignore_index=True)
df_supervised.to_csv('results/supervised_comparison.csv', index=False)

print("\n" + "=" * 85)
print(" ALL 12 ALGORITHMS EXECUTION SUMMARY COMPLETE")
print("=" * 85)

# ---------------------------------------------------------------------------
# 5. GENERATE ACCURACY 100% DEMONSTRATION & GRAND SUMMARY PLOT
# ---------------------------------------------------------------------------
print("\nGenerating Grand Master Summary Visualizations...")

# Grand Plot: 8 Supervised Algorithms Train vs Test Accuracy
fig, ax = plt.subplots(figsize=(14, 7))
bar_width = 0.38
indices = np.arange(len(df_supervised))

b1 = ax.bar(indices - bar_width/2, df_supervised['Train Accuracy (%)'], bar_width,
            label='Training Accuracy (%)', color='#2ecc71', edgecolor='black', alpha=0.9)
b2 = ax.bar(indices + bar_width/2, df_supervised['Test Accuracy (%)'], bar_width,
            label='Testing Accuracy (%)', color='#3498db', edgecolor='black', alpha=0.9)

# Value annotations
for bar in b1:
    h = bar.get_height()
    color = 'darkgreen' if h >= 99.9 else 'black'
    prefix = '[100%] ' if h >= 99.9 else ''
    ax.annotate(f'{prefix}{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                fontsize=9.5, weight='bold', color=color)
for bar in b2:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                fontsize=9.5, weight='bold')

ax.set_xlabel('Supervised Algorithm (Classification & Boosting)', fontsize=12, weight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
ax.set_title('Supervised Learning Benchmark: Training vs Testing Accuracy Across 8 Algorithms\n[100%] Highlights 100.0% Training Accuracy Milestones Achieved',
             fontsize=14, weight='bold', pad=15)
ax.set_xticks(indices)
ax.set_xticklabels([f"{row['Algorithm']}\n({row['Dataset']})" for _, row in df_supervised.iterrows()],
                   fontsize=9.5, rotation=15)
ax.set_ylim(70, 110)
ax.axhline(100, color='red', linestyle='--', linewidth=1.5, label='100% Accuracy Ceiling')

# Category separator
ax.axvline(3.5, color='gray', linestyle=':', linewidth=1.5)
ax.text(1.75, 105, 'Classification (Social Network Ads)', ha='center', fontsize=11, weight='bold', color='#2c3e50')
ax.text(5.75, 105, 'Boosting (Insurance Dataset)', ha='center', fontsize=11, weight='bold', color='#2c3e50')

ax.legend(loc='lower right', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('plots/accuracy_100_percent_demonstration.png', dpi=300)
plt.savefig('plots/master_model_summary.png', dpi=300)
plt.close()

print("Master plot saved: plots/accuracy_100_percent_demonstration.png")
print("All tasks completed successfully!")
