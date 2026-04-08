# 🤖 Model Card — DoorDash Delivery Duration Prediction

> A model card documents what a model does, how it was built, and where it should and shouldn't be trusted.
> This is a modern ML best practice used at Google, Hugging Face, and top tech companies.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | DoorDash Delivery Duration Predictor |
| **Model type** | XGBoost Regressor with Interaction Terms |
| **Version** | v2 (with interaction terms) |
| **Developed by** | Divya Garg |
| **Date** | 2024 |
| **Contact** | [LinkedIn](https://www.linkedin.com/in/divyargarg/) |

---

## Intended Use

**What this model is for:**
- Estimating how long a DoorDash delivery will take in minutes
- Understanding which factors most strongly influence delivery duration
- Portfolio demonstration of end-to-end regression modeling

**Who it is for:**
- Data science learning and portfolio purposes
- Anyone exploring delivery time prediction as a problem

**What this model is NOT for:**
- Production deployment without further validation
- Real-time delivery ETA systems without retraining on current data
- Generalising to other delivery platforms (Uber Eats, Grubhub etc.)

---

## Training Data

| Field | Detail |
|-------|--------|
| **Source** | StrataScratch — DoorDash Historical Delivery Data |
| **Size** | 197,283 orders |
| **Split** | 70% train / 15% validation / 15% test |
| **Time period** | Historical (exact dates unknown) |
| **Geography** | United States (DoorDash markets) |

**Features used (15 total):**

| Feature | Type | Description |
|---------|------|-------------|
| `order_hour` | Numeric | Hour of day order was placed (0–23) |
| `order_month` | Numeric | Month of year (1–12) |
| `order_day_of_week` | Numeric | Day of week (0=Mon, 6=Sun) |
| `is_weekend` | Binary | 1 if Saturday or Sunday |
| `is_rush_hour` | Binary | 1 if hour in [11,12,13,18,19,20] |
| `is_late_night` | Binary | 1 if hour in [22,23,0,1,2] |
| `estimated_order_place_duration` | Numeric | Estimated restaurant prep time (secs) |
| `estimated_store_to_consumer_driving_duration` | Numeric | Estimated drive time (secs) |
| `subtotal` | Numeric | Order value |
| `total_onshift_dashers` | Numeric | Dashers available in the area |
| `total_busy_dashers` | Numeric | Dashers actively on deliveries |
| `hour_x_drive_duration` | Interaction | order_hour × drive duration |
| `hour_x_day` | Interaction | order_hour × day_of_week |
| `dasher_busy_ratio` | Ratio | busy_dashers / total_dashers |
| `order_hour_squared` | Polynomial | Non-linear hour relationship |

---

## Performance

### Test Set Results

| Metric | Score | Meaning |
|--------|-------|---------|
| **MAE** | 11.31 mins | Average prediction error |
| **RMSE** | 15.26 mins | Error weighted toward larger mistakes |
| **R²** | 0.2705 | 27% of variance explained |

### Prediction Accuracy Breakdown

| Tolerance | % of Predictions |
|-----------|-----------------|
| Within 5 minutes | 28.5% |
| Within 10 minutes | 54.8% |
| Within 15 minutes | ~70% (estimated) |

### Baseline Comparison

| Model | Val MAE | Val R² |
|-------|---------|--------|
| Linear Regression | — | — |
| Random Forest | — | — |
| **XGBoost (baseline)** | — | — |
| **XGBoost + interactions (final)** | **11.31** | **0.2705** |

*Fill in baseline values from notebook 04 Cell 17*

---

## Top Feature Importances

| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | `order_hour` | 0.18 | Original |
| 2 | `order_month` | 0.16 | Original |
| 3 | `estimated_order_place_duration` | 0.105 | Original |
| 4 | `order_day_of_week` | 0.102 | Original |
| 5 | `estimated_store_to_consumer_driving_duration` | 0.100 | Original |
| 6 | `hour_x_drive_duration` | — | Interaction ✨ |

---

## Limitations & Known Issues

**What the model cannot see:**
- Real-time traffic conditions
- Weather at time of order
- Restaurant-specific delays or closures
- Dasher GPS location or route efficiency
- Special events (concerts, sports games) that spike demand

**Why R² is 0.27:**
A significant portion of delivery time variance is driven by factors not present in this dataset — live traffic, kitchen wait times, dasher route decisions. This is expected for delivery prediction without real-time operational data.

**Data leakage caught and fixed:**
During feature engineering, `estimation_gap_secs` was accidentally created using the target variable. This caused R² ≈ 1.0 and MAE ≈ 0 — a clear signal of leakage. The column was removed and results returned to realistic levels.

**Temporal drift:**
The model was trained on historical data. Delivery patterns change seasonally and as DoorDash expands to new markets. Periodic retraining would be needed in a production setting.

---

## Ethical Considerations

- No personally identifiable information (PII) was used
- No protected characteristics (race, gender, age) are included in features
- The model predicts time, not people — no direct fairness concerns identified
- Geographic bias may exist if training data over-represents certain cities

---

## How to Use This Model

```python
import pickle
import pandas as pd

# Load model
with open('outputs/models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare input features (must match training feature order)
# See feature list above for required columns
X_new = pd.DataFrame([{
    'order_hour': 19,
    'order_month': 12,
    'order_day_of_week': 4,
    'is_weekend': 0,
    'is_rush_hour': 1,
    # ... all other features
}])

# Predict
predicted_mins = model.predict(X_new)
print(f'Predicted delivery time: {predicted_mins[0]:.1f} minutes')
```

---

*This model card follows the format introduced by Mitchell et al. (2019) — "Model Cards for Model Reporting"*
