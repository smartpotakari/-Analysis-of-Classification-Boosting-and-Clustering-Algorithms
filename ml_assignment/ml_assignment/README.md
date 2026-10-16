# Machine Learning Assignment – Boosting, Clustering, Classification

| # | Task | Dataset | Algorithms trained | Notebook |
|---|------|---------|--------------------|----------|
| 1 | Boosting (regression) | `insurance_pre.csv` – predict insurance `charges` | AdaBoost, Gradient Boosting, Hist Gradient Boosting, XGBoost | `01_Boosting_Insurance.ipynb` |
| 2 | Clustering | `Mall_Customers.csv` – customer segmentation | K-Means, Hierarchical (Ward), DBSCAN, Gaussian Mixture | `02_Clustering_Mall_Customers.ipynb` |
| 3 | Classification | `Social_Network_Ads.csv` – predict `Purchased` | Logistic Regression, KNN, SVM (RBF), Random Forest | `03_Classification_Social_Network_Ads.ipynb` |

## Results

### Clustering (K = 5, features: Annual Income & Spending Score)
| Algorithm | Clusters | Silhouette | Davies-Bouldin |
|---|---|---|---|
| K-Means | 5 | 0.5547 | 0.5722 |
| Hierarchical (Ward) | 5 | 0.5538 | 0.5779 |
| Gaussian Mixture | 5 | 0.5537 | 0.5760 |
| DBSCAN | 4 (+15 noise pts) | 0.4781 | 0.5912 |

### Classification (25% stratified test set)
| Model | Accuracy | F1 | ROC-AUC | 5-fold CV Acc |
|---|---|---|---|---|
| KNN | 0.90 | 0.865 | 0.938 | 0.883 |
| SVM (RBF) | 0.90 | 0.868 | 0.963 | 0.903 |
| Random Forest | 0.89 | 0.849 | 0.925 | 0.840 |
| Logistic Regression | 0.83 | 0.739 | 0.911 | 0.820 |

### Boosting (20% test set) – see the notebook for the full table
Gradient Boosting: Test R² ≈ 0.90, MAE ≈ 2507 | AdaBoost: Test R² ≈ 0.85, MAE ≈ 4245  
(Hist Gradient Boosting and XGBoost are computed when you run the notebook.)

## How to run
```bash
pip install -r requirements.txt
jupyter notebook
```
Or upload the notebooks and the `data/` folder to Google Colab. Run *Kernel → Restart & Run All* before pushing so the outputs are saved in the notebooks.
