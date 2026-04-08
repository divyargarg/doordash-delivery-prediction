# 🚗 DoorDash Delivery Duration Prediction

> Predicting how long a DoorDash delivery will take using historical order data, feature engineering, and machine learning.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Divyargarg-blue?logo=linkedin)](https://www.linkedin.com/in/divyargarg/)

---

## 📌 Project Overview

Every minute matters in food delivery. This project builds a machine learning model to predict the **total delivery duration in minutes** for DoorDash orders — from the moment a customer places an order to when it arrives at their door.

**Core Question:**
> *What factors drive delivery time, and can we accurately predict how long a DoorDash order will take?*

**Dataset:** [StrataScratch — DoorDash Historical Delivery Data](https://www.stratascratch.com/)
**Records:** 197,283 orders

---

## 🎯 Target Variable

```python
total_delivery_duration_mins = actual_delivery_time - created_at
```

This is a **regression problem** — predicting a continuous value (minutes), not a category.

---

## 📊 Model Results

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **MAE** | 11.31 mins | On average predictions are within 11 minutes of actual |
| **RMSE** | 15.26 mins | Larger errors are penalised more heavily |
| **R²** | 0.2705 | Model explains 27% of delivery time variance |

**Best Model:** XGBoost with Interaction Terms

**Prediction accuracy:**
- ✅ 28.5% of predictions land within 5 minutes of actual delivery time
- ✅ 54.8% of predictions land within 10 minutes of actual delivery time

---

## 💡 Key Business Insights

| Finding | Detail |
|---------|--------|
| ⏰ **Time of day is the #1 driver** | `order_hour` is the strongest predictor (importance: 0.18) |
| 🐌 **2pm is the worst time to order** | Hour 14:00 averages **67.6 minutes** — the slowest of any hour |
| 🌅 **5am is the fastest** | Hour 5:00 averages just **40.0 minutes** |
| 🚦 **Rush hour adds ~4 minutes** | Rush hour orders take **3.9 mins longer** on average |
| 📅 **Weekends are slower** | Weekend deliveries take **2.5 mins longer** than weekdays |
| 🛣️ **Distance + timing compounds delay** | `hour_x_drive_duration` was the top interaction term — peak hour combined with long distance is the worst case scenario |

---

## 🗂️ Project Structure

```
doordash-delivery-prediction/
├── data/
│   ├── raw/                          # Original dataset — never modified
│   ├── processed/                    # Cleaned data
│   └── features/                     # Engineered features (v1 + v2 with interactions)
├── notebooks/
│   ├── 01_exploration.ipynb          # EDA — distributions, nulls, correlations
│   ├── 02_cleaning.ipynb             # Null handling, outlier capping, type fixing
│   ├── 03_feature_engineering.ipynb  # Target creation + feature engineering
│   ├── 04_modeling.ipynb             # 12 models compared across 5 families
│   ├── 04b_feature_refinement.ipynb  # Interaction terms + baseline vs improved
│   └── 05_insights.ipynb             # Business insights + storytelling
├── outputs/
│   ├── figures/                      # All charts and visualisations
│   └── models/                       # Saved model files (.pkl)
├── docs/
│   └── decisions.md                  # Every key decision logged with reasoning
├── README.md
└── requirements.txt
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Analysed distributions, null patterns, and correlations across 197,283 records
- Built missing value heatmap and full correlation matrix
- Identified timestamp columns as the primary source of signal

### 2. Data Cleaning
- Removed duplicates and impossible delivery times (< 0 or > 180 mins)
- Imputed nulls — median for numeric, mode for categorical
- Capped outliers using IQR Winsorizing
- **Caught and removed data leakage** — `estimation_gap_secs` was derived from the target and removed before modeling

### 3. Feature Engineering

**Target:**
```python
total_delivery_duration_mins = (actual_delivery_time - created_at).dt.total_seconds() / 60
```

**Time features:** `order_hour`, `order_day_of_week`, `order_month`, `is_weekend`, `is_rush_hour`, `is_late_night`

**Interaction terms (Notebook 04b):**

| Feature | What it captures |
|---------|-----------------|
| `hour_x_drive_duration` | Peak hour + long distance compound effect |
| `hour_x_order_place` | Rush hour hitting slow restaurants |
| `hour_x_day` | Friday 8pm vs Monday 8pm difference |
| `month_x_day` | December Fridays vs July Fridays |
| `dasher_busy_ratio` | True supply constraint (busy / total dashers) |
| `order_hour_squared` | Non-linear U-shaped hour relationship |

### 4. Modeling — 12 Models Across 5 Families

| Family | Models Tested |
|--------|--------------|
| 📐 Linear | Ridge, Lasso, ElasticNet |
| 📍 Instance-Based | KNN Regressor |
| 🌳 Tree | Decision Tree, Random Forest |
| ⚡ Boosting | Gradient Boosting, XGBoost, LightGBM, CatBoost |
| 🤝 Meta-Learners | Voting Ensemble, Stacking Regressor |

**Winner:** XGBoost with interaction terms — trained on 70% of data, evaluated on a held-out 15% test set.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9 | Core language |
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Visualisation |
| Scikit-learn | Preprocessing, models, evaluation |
| XGBoost, LightGBM, CatBoost | Gradient boosting comparison |
| Jupyter Notebooks | Analysis and storytelling |
| Git + GitHub | Version control + portfolio |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/divyargarg/doordash-delivery-prediction.git
cd doordash-delivery-prediction

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run notebooks in order
# 01_exploration → 02_cleaning → 03_feature_engineering → 04_modeling → 04b → 05_insights
```

---

## 📈 What I Would Do Next

- Hyperparameter tuning with Optuna on XGBoost
- Add external features — weather, local events, traffic data
- Explore why R² is 0.27 — investigate missing signal in the data
- Deploy as a REST API using FastAPI + Docker
- Set up model monitoring to detect seasonal drift

---

## 🧠 What I Learned

This project was built as part of an upskilling journey into data science. Key lessons:

- How to structure a professional end-to-end ML project from scratch
- The importance of catching **data leakage** before it fakes your results
- How **interaction terms** capture compound effects single features miss
- The difference between 5 model families and when each is appropriate
- Why **EDA before cleaning** matters — understand before you fix

---

## 👤 Author

**Divya Garg**
[LinkedIn](https://www.linkedin.com/in/divyargarg/) • [GitHub](https://github.com/divyargarg)

---
*Dataset from [StrataScratch](https://www.stratascratch.com/). Built for learning and portfolio purposes.*
