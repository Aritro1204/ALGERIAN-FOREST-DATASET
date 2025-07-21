# 🔥 Algerian Forest Fires Dataset – EDA, Cleaning & Preprocessing

This project focuses on the **exploratory data analysis (EDA)**, **data cleaning**, **feature engineering**, and **preprocessing** of the Algerian Forest Fires dataset. The aim is to understand wildfire patterns and relationships between meteorological conditions and fire occurrences in two Algerian regions.

---

## 📌 Overview

- **Dataset:** Algerian Forest Fires (2012)
- **Instances:** 244 (122 from Bejaia, 122 from Sidi Bel-Abbes)
- **Goal:** Analyze forest fire trends, clean the dataset, engineer features, scale data, and explore patterns for future ML modeling.
- **Tools:** Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn

---

## 📂 Dataset Information

| Feature | Description |
|---------|-------------|
| Date | From June to September 2012 |
| Temperature | Max temp at noon (°C) |
| RH | Relative Humidity (%) |
| Ws | Wind speed (km/h) |
| Rain | Daily rainfall (mm) |
| FFMC, DMC, DC, ISI, BUI, FWI | Fire Weather Index components |
| Classes | Fire / Not Fire (target variable) |

- 🔢 11 input features + 1 output (`Classes`)
- 🗺️ Two regions: Bejaia (0) and Sidi Bel-Abbes (1)

---

## 🧹 Data Cleaning & Preprocessing

- Removed null values
- Fixed column names and whitespace
- Removed unneeded rows (like row 122)
- Created a new column `Region` based on index
- Converted string/object columns to numerical types
- Encoded target classes (`Fire` → 1, `not fire` → 0)

---

## ⚖️ Feature Scaling

- Applied scaling to numerical features to normalize the range for model compatibility.

---
📊 Exploratory Data Analysis (EDA)
🔍 Techniques Used:
Histograms & Density Plots: Visualize feature distributions and detect skewness.

Boxplots: Identify outliers in numerical features.

Correlation Matrix & Heatmap: Explore relationships between variables.

Pie Chart: Understand the distribution of target classes (Fire vs Not Fire).

Monthly Fire Trend Plot: Analyze fire occurrence patterns over different months.

🔥 Key Insights:
FWI, FFMC, and ISI show strong correlation with fire occurrences.

August had the highest number of fire cases in both regions.

The dataset is slightly imbalanced with approximately:

🔥 56% Fire

❌ 44% Not Fire

🧠 Model Training
✅ Models Used:
 Regression

Ridge Regression (RidgeCV)

Lasso Regression (LassoCV)


⚙️ Preprocessing:
Applied StandardScaler to normalize features.

Training performed on X_train_scaled.

📏 Evaluation Metrics:
Accuracy

F1 Score

Confusion Matrix

AUC-ROC Curve

🔁 Cross-Validation
📌 Purpose:
To enhance generalization and avoid overfitting by validating model performance across multiple data splits.

🔄 Techniques Applied:
✅ 1. K-Fold Cross-Validation
Splits data into k equal folds.

Trains on k-1 folds, tests on the remaining fold.

Repeats k times and averages the results.

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print("Cross-validation scores:", scores)
✅ 2. RidgeCV & LassoCV
Automatically selects the best regularization strength (alpha) using internal CV.

Used for better bias-variance tradeoff in regularized regression.

from sklearn.linear_model import RidgeCV

alphas = [0.01, 0.1, 1, 10]
ridge_model = RidgeCV(alphas=alphas, cv=5)
ridge_model.fit(X_train_scaled, y_train)
print("Best alpha:", ridge_model.alpha_)
⚠️ Why Cross-Validation?
📌 Avoids performance overestimation from a single train-test split.

📌 Ensures robust model performance across different data segments.

📌 Helps select optimal hyperparameters.

📦 Libraries Used

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


