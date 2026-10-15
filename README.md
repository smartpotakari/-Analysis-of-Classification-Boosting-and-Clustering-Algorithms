# Machine Learning Benchmark: 12 Algorithms Implementation & Evaluation

A complete Machine Learning project that implements, evaluates, and visualizes **12 essential algorithms** across **Classification**, **Boosting**, and **Clustering** on three real-world datasets.

It runs in **Visual Studio Code** and **Google Colab**, saves every trained model, produces 300 DPI plots, and includes a viva voce preparation guide.

---

## Table of Contents
1. [Project Overview & Dataset Mapping](#1-project-overview--dataset-mapping)
2. [Folder Architecture](#2-folder-architecture)
3. [Environment Setup & Running in VS Code](#3-environment-setup--running-in-vs-code)
4. [Dataset Exploration & Preprocessing](#4-dataset-exploration--preprocessing)
5. [The 12 Algorithms: Theoretical Deep Dive](#5-the-12-algorithms-theoretical-deep-dive)
6. [Experimental Results](#6-experimental-results)
7. [Training Accuracy vs Generalization](#7-training-accuracy-vs-generalization)
8. [Generated Plots & Visualizations Guide](#8-generated-plots--visualizations-guide)
9. [Viva Voce Q&A Preparation](#9-viva-voce-qa-preparation)
10. [Uploading to GitHub](#10-uploading-to-github)

---

## 1. Project Overview & Dataset Mapping

| Category | # | Algorithm | Dataset | Target / Features | Task |
| :--- | :-: | :--- | :--- | :--- | :--- |
| **Classification** | 1 | Decision Tree | `social_networks.csv` | `Purchased` (0/1) | Binary classification |
| | 2 | Random Forest | `social_networks.csv` | `Purchased` (0/1) | Binary classification |
| | 3 | K-Nearest Neighbors (KNN) | `social_networks.csv` | `Purchased` (0/1) | Binary classification |
| | 4 | Support Vector Machine (SVM) | `social_networks.csv` | `Purchased` (0/1) | Binary classification |
| **Boosting** | 5 | AdaBoost | `insurance.csv` | `smoker` (yes/no) | Binary classification |
| | 6 | Gradient Boosting | `insurance.csv` | `smoker` (yes/no) | Binary classification |
| | 7 | XGBoost | `insurance.csv` | `smoker` (yes/no) | Binary classification |
| | 8 | HistGradientBoosting | `insurance.csv` | `smoker` (yes/no) | Binary classification |
| **Clustering** | 9 | K-Means | `mall_customers.csv` | `Annual Income`, `Spending Score` | Unsupervised |
| | 10 | Agglomerative Hierarchical | `mall_customers.csv` | `Annual Income`, `Spending Score` | Unsupervised |
| | 11 | DBSCAN | `mall_customers.csv` | `Annual Income`, `Spending Score` | Density-based |
| | 12 | Gaussian Mixture Model (GMM) | `mall_customers.csv` | `Annual Income`, `Spending Score` | Probabilistic |

---

## 2. Folder Architecture

```text
training/
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignored files & cache
├── README.md                           # Documentation & viva guide
├── ml_12_algorithms_complete.ipynb     # End-to-end notebook (VS Code & Colab)
├── main.py                             # Master pipeline: all 12 algorithms
├── 01_classification.py                # Classification pipeline
├── 02_boosting.py                      # Boosting pipeline
├── 03_clustering.py                    # Clustering pipeline
│
├── social_networks.csv                 # Classification dataset
├── insurance.csv                       # Boosting dataset
├── mall_customers.csv                  # Clustering dataset
│
├── models/                             # Saved trained models (.joblib)
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── k-nearest_neighbors_knn.joblib
│   ├── support_vector_machine_svm.joblib
│   ├── adaboost.joblib
│   ├── gradient_boosting.joblib
│   ├── xgboost.joblib
│   ├── histgradientboosting.joblib
│   ├── kmeans_clustering.joblib
│   ├── agglomerative_clustering.joblib
│   ├── dbscan_clustering.joblib
│   ├── gmm_clustering.joblib
│   └── scalers & label encoders...
│
├── results/                            # CSV evaluation tables
│   ├── classification_results.csv
│   ├── boosting_results.csv
│   ├── clustering_results.csv
│   └── supervised_comparison.csv
│
└── plots/                              # 300 DPI visualizations
    ├── accuracy_100_percent_demonstration.png
    ├── master_model_summary.png
    ├── classification_confusion_matrices.png
    ├── classification_comparison_bar.png
    ├── classification_decision_boundaries.png
    ├── boosting_confusion_matrices.png
    ├── boosting_comparison_bar.png
    ├── boosting_feature_importances.png
    ├── clustering_visualizations.png
    ├── clustering_elbow_silhouette.png
    └── clustering_dendrogram.png
```

---

## 3. Environment Setup & Running in VS Code

### Prerequisites
- Python 3.9+ (3.10 to 3.14 supported)
- Visual Studio Code with the **Python** and **Jupyter** extensions

### Step 1: Open the project folder
In VS Code, choose `File -> Open Folder` (or `Ctrl+K Ctrl+O`) and select the `training` folder.

### Step 2: Install dependencies
Open the integrated terminal and run:
```bash
pip install -r requirements.txt
```
If you prefer to install manually:
```bash
pip install xgboost seaborn scikit-learn pandas numpy matplotlib joblib
```

### Step 3: Run the scripts
```bash
# Full pipeline (all 12 algorithms):
python main.py

# Or run each module individually:
python 01_classification.py
python 02_boosting.py
python 03_clustering.py
```

### Step 4: Run the notebook
Open `ml_12_algorithms_complete.ipynb` in VS Code, pick your Python kernel (top right), and click **Run All**. It also works unchanged in Google Colab.

---

## 4. Dataset Exploration & Preprocessing

### 1. Social Network Ads (`social_networks.csv`)
- **Size**: 400 rows, 5 columns (`User ID`, `Gender`, `Age`, `EstimatedSalary`, `Purchased`)
- **Target**: `Purchased` (0 = did not buy, 1 = purchased)
- **Class balance**: 257 (64.25%) class 0, 143 (35.75%) class 1
- **Preprocessing**:
  - Dropped `User ID` (an arbitrary identifier)
  - Encoded `Gender` numerically with `LabelEncoder`
  - Standardized `Age` and `EstimatedSalary` with `StandardScaler`, fitted on the training split only

### 2. Insurance (`insurance.csv`)
- **Size**: 1,338 rows, 6 columns (`age`, `sex`, `bmi`, `children`, `smoker`, `charges`)
- **Target**: `smoker` (yes/no)
- **Class balance**: 1,064 (79.52%) non-smokers, 274 (20.48%) smokers
- **Preprocessing**:
  - Encoded `sex` and `smoker` with `LabelEncoder`
  - Standardized `age`, `bmi`, and `charges` with `StandardScaler`
- **Note**: `charges` is strongly driven by smoking status, so it is a very informative feature for this target. This is a main reason scores are so high.

### 3. Mall Customers (`mall_customers.csv`)
- **Size**: 200 rows, 5 columns (`CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`)
- **Target**: none (unsupervised)
- **Features used**: `Annual Income (k$)` and `Spending Score (1-100)`, the standard pair for customer segmentation
- **Preprocessing**: standardized with `StandardScaler`

---

## 5. The 12 Algorithms: Theoretical Deep Dive

### Part 1: Classification

#### 1. Decision Tree
- **Mechanism**: Recursively splits the data on feature thresholds that maximize information gain (minimize entropy) or Gini impurity.
- **Entropy**:
  $$\text{Entropy}(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$$
- **Overfitting behavior**: Unpruned (`max_depth=None`), the tree keeps splitting until leaves are pure, so it effectively memorizes the training set.

#### 2. Random Forest
- **Mechanism**: Bagging ensemble of $B$ decision trees, each trained on a bootstrap sample with random feature subsets.
- **Prediction** (majority vote):
  $$\hat{y} = \text{mode}\left( \{ T_b(x) \}_{b=1}^B \right)$$
- **Advantage**: Greatly reduces the variance of a single tree.

#### 3. K-Nearest Neighbors (KNN)
- **Mechanism**: Lazy, instance-based learner. A query point takes the majority class of its $k$ closest training points.
- **Euclidean distance**:
  $$d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$
- **Requirement**: Needs feature scaling so large-valued features (Salary) do not dominate small-valued ones (Age).

#### 4. Support Vector Machine (SVM)
- **Mechanism**: Finds the hyperplane that maximizes the margin between classes.
- **RBF kernel**:
  $$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
- **Advantage**: Produces smooth non-linear boundaries and resists overfitting in low-to-moderate dimensions.

### Part 2: Boosting

#### 5. AdaBoost
- **Mechanism**: Sequentially trains weak learners (usually decision stumps), increasing the weights of misclassified samples after each round.
- **Weight update**:
  $$w_{i}^{(m+1)} = w_i^{(m)} \exp\left( \alpha_m \cdot \mathbb{I}(y_i \neq h_m(x_i)) \right)$$

#### 6. Gradient Boosting
- **Mechanism**: Builds an additive tree model by minimizing a differentiable loss (log-loss). Each tree fits the pseudo-residuals of the current ensemble.
- **Pseudo-residuals**:
  $$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$

#### 7. XGBoost
- **Mechanism**: Optimized, scalable gradient boosting.
- **Key features**:
  - Second-order Taylor expansion (gradients $g_i$ and Hessians $h_i$) for faster convergence
  - Explicit $L_1$ and $L_2$ regularization on leaf weights
  - Cache-aware block structure and sparsity-aware split finding

#### 8. HistGradientBoosting
- **Mechanism**: LightGBM-inspired. Discretizes continuous features into up to 256 integer bins.
- **Advantage**: Split search depends on the number of bins rather than the number of samples, making training much faster on larger data.

### Part 3: Clustering

#### 9. K-Means
- **Mechanism**: Partitions $n$ points into $k$ clusters by minimizing within-cluster sum of squares (inertia):
  $$J = \sum_{j=1}^k \sum_{x_i \in C_j} ||x_i - \mu_j||^2$$
- **Centroid update**: each $\mu_j$ becomes the mean of the points assigned to cluster $j$.

#### 10. Agglomerative Hierarchical Clustering
- **Mechanism**: Bottom-up. Every sample starts as its own cluster; the closest pair is merged repeatedly.
- **Ward's linkage**: Merges the pair that least increases total within-cluster variance. Visualized with a dendrogram.

#### 11. DBSCAN
- **Mechanism**: Finds clusters of arbitrary shape from density.
- **Parameters**: $\epsilon$ (neighborhood radius) and `min_samples` (minimum points for a dense region).
- **Distinction**: Labels outliers as noise (`-1`) instead of forcing them into clusters.

#### 12. Gaussian Mixture Model (GMM)
- **Mechanism**: Generative model assuming data come from a mixture of $k$ Gaussians.
- **Optimization**: Expectation-Maximization (EM) yields soft assignments (posterior probabilities) and parameters $(\mu_k, \Sigma_k, \pi_k)$.

---

## 6. Experimental Results

*All metrics below come from running the code on the datasets (80/20 train/test split).*

### 1. Classification (`social_networks.csv`)
| Algorithm | Train Acc. (%) | Test Acc. (%) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Decision Tree | 99.69 | 88.75 | 0.8125 | 0.8966 | 0.8525 | 0.8895 |
| Random Forest | 99.69 | 90.00 | 0.8387 | 0.8966 | 0.8667 | 0.9341 |
| K-Nearest Neighbors | 91.88 | 90.00 | 0.8387 | 0.8966 | 0.8667 | 0.9304 |
| **Support Vector Machine** | 90.94 | **91.25** | 0.8438 | 0.9310 | 0.8852 | 0.9554 |

### 2. Boosting (`insurance.csv`)
| Algorithm | Train Acc. (%) | Test Acc. (%) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| AdaBoost | 96.82 | 96.27 | 0.9091 | 0.9091 | 0.9091 | 0.9927 |
| **Gradient Boosting** | 100.00 | **97.39** | 0.9286 | 0.9455 | 0.9369 | 0.9961 |
| XGBoost | 100.00 | 97.01 | 0.9273 | 0.9273 | 0.9273 | 0.9949 |
| **HistGradientBoosting** | 100.00 | **97.39** | 0.9138 | 0.9636 | 0.9381 | 0.9940 |

### 3. Clustering (`mall_customers.csv`)
| Algorithm | Clusters | Noise Points | Silhouette | Davies-Bouldin (lower is better) | Calinski-Harabasz |
| :--- | :---: | :---: | :---: | :---: | :---: |
| K-Means | 5 | 0 | 0.5547 | 0.5722 | **248.65** |
| Agglomerative (Ward) | 5 | 0 | 0.5538 | 0.5779 | 244.41 |
| DBSCAN | 6 | 23 | **0.5577** | **0.5106** | 245.36 |
| GMM | 5 | 0 | 0.5537 | 0.5760 | 244.94 |

DBSCAN's scores are not directly comparable to the others: it leaves 23 points unclustered as noise, and it finds 6 groups instead of 5.

---

## 7. Training Accuracy vs Generalization

100% training accuracy is **not** a performance milestone. It shows that a model has enough capacity to fit the training data, and it can also signal overfitting. The number that matters is **test accuracy** on unseen data.

| Algorithm | Train Acc. | Test Acc. | Gap |
| :--- | :---: | :---: | :---: |
| Gradient Boosting | 100.00% | 97.39% | 2.61 pts |
| XGBoost | 100.00% | 97.01% | 2.99 pts |
| HistGradientBoosting | 100.00% | 97.39% | 2.61 pts |
| Decision Tree | 99.69% | 88.75% | 10.94 pts |
| Random Forest | 99.69% | 90.00% | 9.69 pts |
| SVM | 90.94% | 91.25% | -0.31 pts |

### Why Decision Tree and Random Forest stop at 99.69%
`social_networks.csv` contains **contradictory duplicates**, rows with identical features but different labels:
- Row 59: `[Female, Age 32, Salary 117000] -> Purchased = 0`
- Row 182: `[Female, Age 32, Salary 117000] -> Purchased = 1`

No deterministic function can output both labels for the same input. With 320 training samples, the ceiling is therefore $319 / 320 = 99.6875\% \approx 99.69\%$.

### Why the boosting models reach 100% training accuracy
On `insurance.csv`, the boosted trees have enough capacity to separate smokers from non-smokers perfectly on the training set. This is helped by `charges` being a very strong predictor of smoking status. Their test accuracy (97%+) and ROC-AUC (above 0.99) show they still generalize well.

### Reading the gaps
- Decision Tree and Random Forest show a gap of roughly 10 points, which is clear overfitting on a small (320-row) training set.
- The boosting models have a gap under 3 points, so their fit is healthy.
- SVM has almost no gap, so it generalizes best on the Social Network Ads data.
- Results come from a single train/test split; cross-validation would give more reliable estimates.

---

## 8. Generated Plots & Visualizations Guide

All plots are saved at **300 DPI** in `plots/`:

| Filename | Description |
| :--- | :--- |
| `accuracy_100_percent_demonstration.png` | Train vs test accuracy for all 8 supervised models |
| `master_model_summary.png` | Summary view across all models |
| `classification_confusion_matrices.png` | 2x2 grid of confusion matrices for the 4 classifiers |
| `classification_comparison_bar.png` | Train vs test accuracy for the classifiers |
| `classification_decision_boundaries.png` | 2D decision boundaries over Age and Salary |
| `boosting_confusion_matrices.png` | 2x2 grid of confusion matrices for the boosting models |
| `boosting_comparison_bar.png` | Train vs test accuracy for the boosting models |
| `boosting_feature_importances.png` | Feature importances (`charges` and `bmi` dominate) |
| `clustering_visualizations.png` | Scatter plots of customer clusters per algorithm |
| `clustering_elbow_silhouette.png` | Inertia and silhouette curves for K=2 to 10 |
| `clustering_dendrogram.png` | Ward's linkage dendrogram cut at 5 clusters |

---

## 9. Viva Voce Q&A Preparation

### General & Conceptual

**Q1: What is the difference between supervised and unsupervised learning?**
- **Supervised**: learns from labeled data $(X, y)$ to predict known targets (classification, boosting).
- **Unsupervised**: finds structure in unlabeled data $(X)$, such as groups or dense regions (clustering).

**Q2: Why evaluate clustering with Silhouette Score instead of accuracy?**
- There are no ground-truth labels, so accuracy is undefined. The silhouette score measures compactness and separation:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean distance to the nearest other cluster.

**Q3: What is data leakage, and how was it prevented?**
- Leakage is when information from outside the training data influences the model. Here:
  1. Data were split into train and test sets *before* scaling.
  2. `StandardScaler` was fitted only on `X_train`.
  3. The fitted scaler was applied to `X_test` with `transform`.
  4. The arbitrary `User ID` column was dropped.

**Q4: What is the bias-variance tradeoff?**
- **Bias**: error from overly simple assumptions (underfitting).
- **Variance**: error from sensitivity to the training sample (overfitting, e.g. an unpruned tree).
- Random Forests reduce variance through bagging; boosting reduces bias through sequential residual fitting.

### Algorithm-Specific

**Q5: Why did the boosting models reach 100% training accuracy, and is that good?**
- They have enough capacity to drive the log-loss to zero on the training set, aided by the very informative `charges` feature. On its own it is not an achievement; what matters is test accuracy (97%+) and ROC-AUC (above 0.99), which show good generalization.

**Q6: Why can't the Decision Tree reach 100% training accuracy on Social Network Ads?**
- Two rows have identical features but opposite labels (rows 59 and 182), so one of them must be misclassified. The ceiling is 319/320 = 99.69%.

**Q7: What is the difference between bagging and boosting?**
- **Bagging** (Random Forest): trains trees in parallel on bootstrap samples and votes, reducing variance.
- **Boosting** (AdaBoost, XGBoost): trains trees in sequence, each correcting the previous errors, reducing bias.

**Q8: How does DBSCAN handle noise compared to K-Means?**
- K-Means assigns every point to a cluster. DBSCAN labels points that lack `min_samples` neighbors within $\epsilon$ as noise (`-1`).

**Q9: What is the difference between K-Means and GMM?**
- **K-Means**: hard assignment by Euclidean distance to centroids, implicitly spherical clusters.
- **GMM**: soft assignment (probabilities) with full covariance matrices, allowing elliptical clusters of different sizes and orientations.

**Q10: How was K=5 chosen for the Mall Customers data?**
- The elbow curve flattens at K=5 and the silhouette score peaks there (`clustering_elbow_silhouette.png`). The dendrogram also supports a 5-cluster cut.

---

## 10. Uploading to GitHub

### Step 1: Initialize the repository
```bash
cd path/to/training
git init
```

### Step 2: Configure Git (if not already set)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 3: Commit the files
```bash
git add .
git commit -m "Add ML benchmark: 12 algorithms with models, results, and plots"
```

### Step 4: Create the GitHub repository
1. Log in to [GitHub](https://github.com).
2. Click **+** (top right) and choose **New repository**.
3. Name it (for example `machine-learning-12-algorithms-benchmark`).
4. Choose **Public** or **Private**.
5. Do **not** tick "Initialize with a README" (this project already has one).
6. Click **Create repository**.

### Step 5: Push
```bash
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
git push -u origin main
```

---

*Machine Learning academic benchmark project.*
