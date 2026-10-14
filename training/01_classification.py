"""
=============================================================================
PART 1: CLASSIFICATION PIPELINE (4 ALGORITHMS)
=============================================================================
Dataset: Social Network Ads (social_networks.csv)
Algorithms:
  1. Decision Tree Classifier
  2. Random Forest Classifier
  3. K-Nearest Neighbors (KNN)
  4. Support Vector Machine (SVM)
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 80)
print(" 1. DATA EXPLORATION: SOCIAL NETWORK ADS DATASET")
print("=" * 80)

# 1. Load Data
df = pd.read_csv('social_networks.csv')
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("Columns and Data Types:")
print(df.dtypes)
print("\nMissing Values per Column:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe().round(2))

print("\nTarget Class Distribution ('Purchased'):")
print(df['Purchased'].value_counts())
print(f"Class Balance: {df['Purchased'].value_counts(normalize=True).round(4).to_dict()}\n")

# Check for duplicate feature combinations with conflicting labels
feature_cols = ['Gender', 'Age', 'EstimatedSalary']
conflicting = df[df.duplicated(subset=feature_cols, keep=False)]
print(f"Total duplicate feature rows: {len(conflicting)}")
if len(conflicting) > 0:
    print("Sample duplicate feature rows (demonstrating real-world label noise):")
    print(conflicting.sort_values(by=['Age', 'EstimatedSalary']).head(4))

print("\n" + "=" * 80)
print(" 2. DATA PREPROCESSING & TRAIN/TEST SPLIT")
print("=" * 80)

# Encode Gender
le_gender = LabelEncoder()
df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])

# Features & Target
# We use Age, EstimatedSalary, and Gender_Encoded
X = df[['Age', 'EstimatedSalary', 'Gender_Encoded']].copy()
y = df['Purchased'].copy()

# Stratified Train/Test split to avoid data leakage
# RS 79 or 42 are standard reproducible seeds
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Feature Scaling: fit scaler ONLY on training data to prevent data leakage!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessors for inference
joblib.dump(scaler, 'models/scaler_classification.joblib')
joblib.dump(le_gender, 'models/encoder_gender.joblib')

print("\n" + "=" * 80)
print(" 3. TRAINING & EVALUATION OF 4 CLASSIFICATION ALGORITHMS")
print("=" * 80)

# Define models
models = {
    'Decision Tree': {
        'model': DecisionTreeClassifier(criterion='entropy', random_state=42),
        'scaled': False
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42),
        'scaled': False
    },
    'K-Nearest Neighbors (KNN)': {
        'model': KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
        'scaled': True
    },
    'Support Vector Machine (SVM)': {
        'model': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'scaled': True
    }
}

results_list = []
cms = {}

for name, cfg in models.items():
    clf = cfg['model']
    use_scaled = cfg['scaled']
    
    # Select appropriate features (tree algorithms don't require scaling, distance models do)
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled if use_scaled else X_test
    
    # Train
    clf.fit(X_tr, y_train)
    
    # Predict
    y_train_pred = clf.predict(X_tr)
    y_test_pred = clf.predict(X_te)
    y_test_proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, 'predict_proba') else None
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else np.nan
    cm = confusion_matrix(y_test, y_test_pred)
    cms[name] = cm
    
    # Save model
    filename = f"models/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
    joblib.dump(clf, filename)
    
    results_list.append({
        'Algorithm': name,
        'Train Accuracy (%)': round(train_acc * 100, 2),
        'Test Accuracy (%)': round(test_acc * 100, 2),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4) if not np.isnan(roc_auc) else 'N/A'
    })
    
    print(f"\n>>> {name.upper()} RESULTS:")
    print(f"  Training Accuracy: {train_acc * 100:.2f}%  {'[100% PERFECT FIT MILESTONE]' if train_acc >= 0.999 else ''}")
    print(f"  Testing Accuracy:  {test_acc * 100:.2f}%")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall:            {rec:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

# Save results to CSV
df_results = pd.DataFrame(results_list)
df_results.to_csv('results/classification_results.csv', index=False)

print("\n" + "=" * 80)
print(" CLASSIFICATION COMPARISON TABLE")
print("=" * 80)
print(df_results.to_string(index=False))

# ---------------------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------------------
print("\nGenerating Classification Visualization Plots...")

# Plot 1: 4 Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, cm) in enumerate(cms.items()):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'},
                xticklabels=['Not Purchased (0)', 'Purchased (1)'],
                yticklabels=['Not Purchased (0)', 'Purchased (1)'])
    acc_val = df_results.loc[df_results['Algorithm'] == name, 'Test Accuracy (%)'].values[0]
    ax.set_title(f'{name}\nTest Accuracy: {acc_val:.1f}%', fontsize=13, weight='bold', pad=10)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

plt.suptitle('Confusion Matrices — 4 Classification Algorithms (Social Network Ads)',
             fontsize=15, weight='bold', y=0.99)
plt.tight_layout()
plt.savefig('plots/classification_confusion_matrices.png', dpi=300)
plt.close()

# Plot 2: Algorithm Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
indices = np.arange(len(df_results))

bars1 = ax.bar(indices - bar_width/2, df_results['Train Accuracy (%)'], bar_width,
               label='Train Accuracy (%)', color='#2ecc71', edgecolor='black', alpha=0.85)
bars2 = ax.bar(indices + bar_width/2, df_results['Test Accuracy (%)'], bar_width,
               label='Test Accuracy (%)', color='#3498db', edgecolor='black', alpha=0.85)

# Add values on top of bars
for bar in bars1:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                fontsize=10, weight='bold')
for bar in bars2:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                fontsize=10, weight='bold')

ax.set_xlabel('Classification Algorithm', fontsize=12, weight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
ax.set_title('Classification Performance: Train vs Test Accuracy\n(Demonstrating 100% Training Fidelity on Trees)',
             fontsize=14, weight='bold', pad=15)
ax.set_xticks(indices)
ax.set_xticklabels(df_results['Algorithm'], fontsize=11, rotation=10)
ax.set_ylim(70, 108)
ax.axhline(100, color='red', linestyle='--', alpha=0.7, label='100% Milestone Target')
ax.legend(loc='lower right', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('plots/classification_comparison_bar.png', dpi=300)
plt.close()

# Plot 3: 2D Decision Boundaries for the 4 Classifiers (using Age vs EstimatedSalary)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# Fit 2D models for boundary visualization
X_2d = df[['Age', 'EstimatedSalary']].values
y_2d = df['Purchased'].values
scaler_2d = StandardScaler()
X_2d_scaled = scaler_2d.fit_transform(X_2d)

x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

models_2d = [
    ('Decision Tree', DecisionTreeClassifier(criterion='entropy', random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('K-Nearest Neighbors (KNN)', KNeighborsClassifier(n_neighbors=5)),
    ('Support Vector Machine (SVM)', SVC(kernel='rbf', C=1.0, random_state=42))
]

for idx, (m_name, m_obj) in enumerate(models_2d):
    ax = axes[idx]
    m_obj.fit(X_2d_scaled, y_2d)
    Z = m_obj.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = ax.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], c=y_2d,
                         cmap=plt.cm.coolwarm, edgecolors='k', s=35, alpha=0.85)
    ax.set_title(f'{m_name} Decision Boundary', fontsize=12, weight='bold')
    ax.set_xlabel('Age (Standardized)', fontsize=10)
    ax.set_ylabel('Estimated Salary (Standardized)', fontsize=10)

plt.suptitle('Decision Boundaries of 4 Classification Algorithms (Social Network Ads)',
             fontsize=15, weight='bold', y=0.99)
plt.tight_layout()
plt.savefig('plots/classification_decision_boundaries.png', dpi=300)
plt.close()

print("All classification models, results, and plots successfully saved!")
