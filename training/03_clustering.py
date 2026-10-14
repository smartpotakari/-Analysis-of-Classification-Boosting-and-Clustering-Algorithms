"""
=============================================================================
PART 3: CLUSTERING PIPELINE (4 ALGORITHMS)
=============================================================================
Dataset: Mall Customers (mall_customers.csv)
Algorithms (Unsupervised Learning — No target labels used):
  1. K-Means Clustering
  2. Agglomerative Hierarchical Clustering
  3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  4. Gaussian Mixture Model (GMM)
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import scipy.cluster.hierarchy as sch

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 80)
print(" 1. DATA EXPLORATION: MALL CUSTOMERS DATASET")
print("=" * 80)

# 1. Load Data
df = pd.read_csv('mall_customers.csv')
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("Columns and Data Types:")
print(df.dtypes)
print("\nMissing Values per Column:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe().round(2))

print("\n" + "=" * 80)
print(" 2. FEATURE SELECTION & PREPROCESSING")
print("=" * 80)

# Features: Annual Income (k$) and Spending Score (1-100)
# (In unsupervised learning, CustomerID is an identifier and labels are NOT used)
feature_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[feature_cols].copy()

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler_clustering.joblib')

print(f"Clustering Features: {feature_cols}")
print(f"Feature matrix shape: {X_scaled.shape}")

# ---------------------------------------------------------------------------
# ELBOW METHOD & SILHOUETTE ANALYSIS FOR OPTIMAL K
# ---------------------------------------------------------------------------
print("\nCalculating Elbow Method and Silhouette Scores for K=2 to 10...")
k_range = range(2, 11)
inertias = []
sil_scores = []

for k in k_range:
    km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_test.fit(X_scaled)
    inertias.append(km_test.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km_test.labels_))

# Plot Elbow and Silhouette curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow curve
ax1.plot(list(k_range), inertias, marker='o', linewidth=2.5, color='#e74c3c')
ax1.set_title('Elbow Method (Inertia vs K)', fontsize=13, weight='bold')
ax1.set_xlabel('Number of Clusters (K)', fontsize=11)
ax1.set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=11)
ax1.axvline(5, color='gray', linestyle='--', label='Elbow Point (K=5)')
ax1.legend()

# Silhouette curve
ax2.plot(list(k_range), sil_scores, marker='s', linewidth=2.5, color='#2980b9')
ax2.set_title('Silhouette Analysis across K', fontsize=13, weight='bold')
ax2.set_xlabel('Number of Clusters (K)', fontsize=11)
ax2.set_ylabel('Silhouette Score', fontsize=11)
best_k = list(k_range)[np.argmax(sil_scores)]
ax2.axvline(best_k, color='green', linestyle='--', label=f'Optimal Peak (K={best_k})')
ax2.legend()

plt.suptitle('Determining Optimal Number of Clusters (Mall Customers)', fontsize=15, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/clustering_elbow_silhouette.png', dpi=300)
plt.close()

print(f"Optimal cluster count determined: K = {best_k} (Highest Silhouette Score: {max(sil_scores):.4f})")

print("\n" + "=" * 80)
print(" 3. TRAINING & EVALUATION OF 4 CLUSTERING ALGORITHMS")
print("=" * 80)

# Optimal K = 5
k_optimal = 5

# 1. K-Means
km = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42, n_init=10)
labels_km = km.fit_predict(X_scaled)
joblib.dump(km, 'models/kmeans_clustering.joblib')

# 2. Agglomerative Hierarchical Clustering
agg = AgglomerativeClustering(n_clusters=k_optimal, metric='euclidean', linkage='ward')
labels_agg = agg.fit_predict(X_scaled)
joblib.dump(agg, 'models/agglomerative_clustering.joblib')

# 3. DBSCAN
db = DBSCAN(eps=0.35, min_samples=5)
labels_db = db.fit_predict(X_scaled)
joblib.dump(db, 'models/dbscan_clustering.joblib')

# 4. Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=k_optimal, covariance_type='full', random_state=42)
labels_gmm = gmm.fit_predict(X_scaled)
joblib.dump(gmm, 'models/gmm_clustering.joblib')

clustering_models = {
    'K-Means': labels_km,
    'Agglomerative Hierarchical': labels_agg,
    'DBSCAN': labels_db,
    'Gaussian Mixture Model (GMM)': labels_gmm
}

results_list = []

for name, labels in clustering_models.items():
    # Handle noise (-1) in DBSCAN for silhouette calculation
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = list(labels).count(-1)
    
    # Calculate metrics
    if n_clusters > 1:
        # If noise exists, compute silhouette on non-noise points
        if -1 in unique_labels:
            non_noise_mask = (labels != -1)
            sil = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
            db_index = davies_bouldin_score(X_scaled[non_noise_mask], labels[non_noise_mask])
            ch_index = calinski_harabasz_score(X_scaled[non_noise_mask], labels[non_noise_mask])
        else:
            sil = silhouette_score(X_scaled, labels)
            db_index = davies_bouldin_score(X_scaled, labels)
            ch_index = calinski_harabasz_score(X_scaled, labels)
    else:
        sil, db_index, ch_index = np.nan, np.nan, np.nan
        
    results_list.append({
        'Algorithm': name,
        'Clusters Found': n_clusters,
        'Noise Points': n_noise,
        'Silhouette Score': round(sil, 4) if not np.isnan(sil) else 'N/A',
        'Davies-Bouldin Index': round(db_index, 4) if not np.isnan(db_index) else 'N/A',
        'Calinski-Harabasz Index': round(ch_index, 2) if not np.isnan(ch_index) else 'N/A'
    })
    
    print(f"\n>>> {name.upper()} RESULTS:")
    print(f"  Clusters Found:          {n_clusters}")
    print(f"  Noise Points (if any):   {n_noise}")
    print(f"  Silhouette Score:        {sil:.4f}  (Higher is better, >0.5 indicates strong structure)")
    print(f"  Davies-Bouldin Index:    {db_index:.4f}  (Lower is better)")
    print(f"  Calinski-Harabasz Index: {ch_index:.2f}  (Higher is better)")

# Save results to CSV
df_results = pd.DataFrame(results_list)
df_results.to_csv('results/clustering_results.csv', index=False)

print("\n" + "=" * 80)
print(" CLUSTERING COMPARISON TABLE")
print("=" * 80)
print(df_results.to_string(index=False))

# ---------------------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------------------
print("\nGenerating Clustering Visualization Plots...")

# Plot 1: 2x2 Cluster Visualization Grid
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

cluster_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#34495e']

for idx, (name, labels) in enumerate(clustering_models.items()):
    ax = axes[idx]
    unique_lbls = sorted(list(set(labels)))
    
    for lbl in unique_lbls:
        if lbl == -1:
            # Noise point in DBSCAN
            mask = (labels == lbl)
            ax.scatter(X.loc[mask, 'Annual Income (k$)'],
                       X.loc[mask, 'Spending Score (1-100)'],
                       c='#7f8c8d', marker='x', s=45, label='Noise')
        else:
            mask = (labels == lbl)
            color = cluster_colors[lbl % len(cluster_colors)]
            ax.scatter(X.loc[mask, 'Annual Income (k$)'],
                       X.loc[mask, 'Spending Score (1-100)'],
                       c=color, s=55, alpha=0.85, edgecolors='black',
                       linewidth=0.5, label=f'Cluster {lbl+1}')
            
    # Plot centroids for K-Means
    if name == 'K-Means':
        centroids_orig = scaler.inverse_transform(km.cluster_centers_)
        ax.scatter(centroids_orig[:, 0], centroids_orig[:, 1],
                   s=220, c='yellow', marker='*', edgecolors='black',
                   linewidth=1.5, label='Centroids')
        
    sil_val = df_results.loc[df_results['Algorithm'] == name, 'Silhouette Score'].values[0]
    ax.set_title(f'{name}\nSilhouette Score: {sil_val}', fontsize=12, weight='bold', pad=8)
    ax.set_xlabel('Annual Income (k$)', fontsize=10, weight='bold')
    ax.set_ylabel('Spending Score (1-100)', fontsize=10, weight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

plt.suptitle('Cluster Visualizations — 4 Clustering Algorithms (Mall Customers)',
             fontsize=15, weight='bold', y=0.99)
plt.tight_layout()
plt.savefig('plots/clustering_visualizations.png', dpi=300)
plt.close()

# Plot 2: Dendrogram for Hierarchical Clustering
print("Generating Hierarchical Clustering Dendrogram...")
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'),
                            truncate_mode='lastp', p=30,
                            leaf_rotation=90, leaf_font_size=10,
                            show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14, weight='bold', pad=12)
plt.xlabel('Customer Cluster Index / Size', fontsize=11, weight='bold')
plt.ylabel('Euclidean Distance (Ward Linkage)', fontsize=11, weight='bold')
plt.axhline(y=7.5, color='r', linestyle='--', label='Cut Threshold for 5 Clusters')
plt.legend()
plt.tight_layout()
plt.savefig('plots/clustering_dendrogram.png', dpi=300)
plt.close()

print("All clustering models, results, and plots successfully saved!")
