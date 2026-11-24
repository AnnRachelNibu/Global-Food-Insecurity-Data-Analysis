# Exploring Patterns of Food Insecurity and Malnutrition through Data Mining and Risk Classification

**Authors:** Zobiya Fathima, Ann Rachel Nibu

## Overview
Malnutrition and food insecurity disproportionately affect children under five. This
project mines the WHO Global Database on Child Growth and Malnutrition to find patterns
of child malnutrition. It uses BIRCH clustering to group records into three risk levels
(**High, Moderate, Low**), then trains classifiers to predict risk level from
demographic and socioeconomic factors.

## Objectives
- Preprocess a large, multi-country child malnutrition dataset
- Cluster records into risk levels using nutrition indicators
- Compare clustering algorithms with internal validation metrics
- Predict risk level from country, age, sex, mother's education, wealth and urban/rural residence
- Compare five classifiers and identify the best performer

## Dataset
- **Source:** WHO Global Database on Child Growth and Malnutrition
- **Size:** about 43,000 records (country x survey year x population group)
- **Indicators:** Stunting, Wasting, Severe wasting, Underweight, Overweight (%)
- **Other columns used:** Country, Median Year, Age, Sex, Mother education, Wealth quintile, Urban/Rural
- **File:** `food insecurity.xlsx`

## Methodology
![Framework](images/framework.png)

### 1. Preprocessing
- Removed irrelevant columns (author, notes, reference title, sample size, region, etc.)
- Dropped rows with more than 50% missing values
- **KNN imputation** (k=5) for Mother education and Wealth quintile
- **Label encoding** of Sex, Mother education, Wealth quintile, Urban/Rural
- Converted age ranges into numeric midpoints

### 2. Clustering
Features: Wasting, Severe wasting, Stunting, Underweight (standardised).
BIRCH parameters: `n_clusters=3, branching_factor=50, threshold=0.5`.

| Algorithm | Silhouette (higher is better) | Davies-Bouldin (lower is better) | Calinski-Harabasz (higher is better) |
|---|---|---|---|
| **BIRCH** | **0.578** | **0.575** | 3971 |
| KMeans | 0.370 | 1.042 | 32173 |
| MiniBatch-KMeans | 0.350 | 1.053 | 31313 |
| GMM | 0.255 | 1.509 | 13161 |
| FAISS | 0.426 | 0.837 | 56342 |

BIRCH gave the best Silhouette and Davies-Bouldin scores and was used to create the labels.
Clusters were mapped to High / Moderate / Low Risk by their mean malnutrition values.

![BIRCH clusters](images/birch_pca.png)

### 3. Classification
- **Features:** Country, Median Year, Age, Gender, Mother's education, Wealth, Rural/Urban (one-hot encoded)
- **Target:** BIRCH-derived risk label
- **Split:** 80/20 stratified (8,537 test records)
- The clustering indicators are *not* used as classifier inputs, which avoids target leakage.
- XGBoost and Random Forest tuned with `RandomizedSearchCV` (20 iterations, 3-fold CV)

**Best XGBoost parameters:** `n_estimators=200, max_depth=10, learning_rate=0.2,
subsample=1.0, colsample_bytree=0.6, gamma=0.1, min_child_weight=5`

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **XGBoost** | **0.86** | **0.86** | **0.86** | **0.86** |
| Random Forest | 0.85 | 0.85 | 0.85 | 0.85 |
| KNN | 0.82 | 0.82 | 0.82 | 0.82 |
| Gradient Boosting | 0.80 | 0.79 | 0.80 | 0.77 |
| Logistic Regression | 0.77 | 0.75 | 0.77 | 0.75 |

(Precision, recall and F1 are weighted averages.)

### XGBoost per-class performance

| Risk Level | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| High Risk | 0.83 | 0.69 | 0.75 | 520 |
| Low Risk | 0.90 | 0.94 | 0.92 | 5,812 |
| Moderate Risk | 0.77 | 0.70 | 0.73 | 2,205 |
| **Macro avg** | 0.83 | 0.78 | 0.80 | 8,537 |

One-vs-rest ROC AUC: **High 0.98, Low 0.95, Moderate 0.93**.

![Confusion matrix](images/confusion_matrix.png)
![ROC curves](images/roc.png)

### Key findings
- XGBoost performed best, followed closely by Random Forest. Both ensemble methods clearly beat Logistic Regression, which suggests a non-linear relationship between demographics and risk.
- Low Risk is the largest class (about 68% of the data) and the easiest to predict.
- Moderate Risk is hardest to separate because it overlaps with both extremes.
- High Risk is the smallest class (about 6%), but the model still separates it well (AUC 0.98).

## Tech stack
Python 3, pandas, NumPy, scikit-learn, XGBoost, FAISS, Matplotlib, Seaborn. Run on Google Colab (CPU).

## Limitations and future work
- Risk labels come from unsupervised clustering, not official WHO thresholds
- Class imbalance reduces recall for the High and Moderate classes
- Future work: rainfall anomalies, food price inflation, supply chain data, time-series forecasting, SHAP feature importance
