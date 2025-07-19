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

## 📊 Exploratory Data Analysis (EDA)

### 🔍 Techniques Used:
- **Histograms & Density Plots:** To study feature distributions and skewness
- **Boxplots:** To detect outliers
- **Correlation Matrix & Heatmap:** To identify relationships between features
- **Pie Chart:** To visualize class distribution
- **Monthly Analysis Plot:** To examine seasonal fire trends

### 🔥 Insights:
- Fire Weather Index (FWI), FFMC, and ISI are highly correlated with fire occurrence
- **August** showed the highest fire activity in both regions
- Class imbalance observed: ~56% fire, ~44% not fire

---

## 📦 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
