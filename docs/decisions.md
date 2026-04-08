# 📝 Decision Log — DoorDash Delivery Prediction

> Every significant decision made during this project, with reasoning.
> This file is updated as the project evolves.

---

## 📅 Project Setup

**Decision:** Create separate notebooks for each stage (EDA, cleaning, features, modeling, insights)
**Reason:** Each stage has a different goal. Keeping them separate means you can re-run any step without affecting others, and the flow is easy for anyone reading the project to follow.

**Decision:** Use `data/raw/` as sacred — never write to it
**Reason:** The original dataset is your source of truth. If anything goes wrong downstream, you can always start fresh from raw data.

---

## 🔍 EDA Decisions

**Decision:** Run EDA before cleaning, not after
**Reason:** You need to understand the data as it actually is — warts and all — before deciding how to fix it. Cleaning first would hide information you need to make good decisions.

---

## 🧹 Cleaning Decisions

**Decision:** Cap outliers using IQR Winsorizing instead of dropping rows
**Reason:** Dropping rows removes data permanently. Capping at the IQR boundary keeps the row in the dataset but removes the extreme distortion. Better for a large dataset where every row has value.

**Decision:** Use median imputation for numeric nulls, mode for categorical
**Reason:** The target variable (delivery duration) had a right-skewed distribution. Mean imputation would inflate values. Median is more robust to skew.

**Decision:** Remove rows where `total_delivery_duration_mins` < 0 or > 180
**Reason:** Negative delivery time is physically impossible — the order would have arrived before it was placed. Deliveries over 3 hours are almost certainly data entry errors, not real orders. Only 0.X% of rows were affected.

---

## ⚙️ Feature Engineering Decisions

**Decision:** Create target as `actual_delivery_time - created_at` in minutes
**Reason:** This is the most direct measure of what a customer experiences — total wait time from order placement to delivery. Seconds were converted to minutes for interpretability.

**Decision:** Create `is_rush_hour` as binary flag for hours [11,12,13,18,19,20]
**Reason:** Lunch (11am–1pm) and dinner (6pm–8pm) are the two well-known demand peaks in food delivery. A binary flag is cleaner than relying on raw hour alone for this specific signal.

**Decision:** Add `order_hour_squared` as a polynomial feature
**Reason:** The relationship between hour and delivery time is not linear. Deliveries at noon and 7pm are both slow, but midnight is fast again. A squared term lets the model learn this curved U-shaped relationship.

**Decision:** Drop `estimation_gap_secs` after creating it
**Reason:** This column was calculated as `total_delivery_duration - estimated_duration` — it directly contains the target variable inside it. Keeping it would be **data leakage** and produce artificially perfect model results (R² = 1.0, MAE ≈ 0). Caught and removed before modeling.

**Decision:** Drop collinear features with correlation > 0.85
**Reason:** Two highly correlated features tell the model the same thing twice. This adds noise and can destabilise linear models. Keeping the more interpretable of the two.

---

## 🔗 Interaction Term Decisions (Notebook 04b)

**Decision:** Create `hour_x_drive_duration` as top priority interaction
**Reason:** From feature importance chart, `order_hour` (0.18) and `estimated_store_to_consumer_driving_duration` (0.10) were both top-5 features. Their interaction captures the worst-case scenario: a long delivery distance during peak hours — a compound effect neither feature captures alone.

**Decision:** Create `dasher_busy_ratio` instead of raw dasher count
**Reason:** Raw `total_onshift_dashers` count is misleading. 10 dashers with 100 active orders is very different from 10 dashers with 5 orders. The ratio (busy / total) captures true supply constraint more accurately.

**Decision:** Keep interaction terms in a separate notebook (04b) rather than editing 04
**Reason:** Notebook 04 is the baseline. Overwriting it removes the ability to compare before and after. 04b allows a clean side-by-side comparison showing whether interaction terms actually helped.

---

## 🤖 Modeling Decisions

**Decision:** Use 70/15/15 train/validation/test split instead of 80/20
**Reason:** With 197,283 records, a 15% test set gives ~29,000 samples — large enough for reliable evaluation. The validation set is used for model selection; the test set is only touched once at the very end.

**Decision:** Fit StandardScaler on train set only, then transform val and test
**Reason:** Fitting the scaler on all data would let it "see" the test set's distribution during training — a form of data leakage. Fitting only on train data ensures the test set is truly unseen.

**Decision:** Test 12 models across 5 families before picking a winner
**Reason:** No model is universally best. Testing across families (linear, tree, boosting, ensemble) gives real evidence for which approach suits this dataset, rather than jumping to XGBoost by assumption.

**Decision:** Use MAE as the primary metric, not RMSE or R²
**Reason:** MAE is directly interpretable — it means "on average, the model is off by X minutes." RMSE penalises large errors more heavily (useful but less intuitive). R² shows relative performance but is harder to explain to a non-technical audience.

**Decision:** Choose XGBoost with interaction terms as final model
**Reason:** XGBoost consistently outperformed linear models and matched or beat other boosting libraries on this dataset. Adding interaction terms further improved MAE. The model is well-understood, widely deployed in industry, and has good explainability through feature importance.

---

## 📊 Results Context

**R² of 0.27 — what does this mean?**
The model explains 27% of the variance in delivery time. This is not a high R², but it is expected for delivery prediction where many factors are unobserved — restaurant-level delays, traffic conditions, dasher behaviour, weather. The features available in this dataset capture time and order signals but not real-time operational state. Adding external data (weather, live traffic) would likely improve this significantly.

---

## 🔜 Decisions Still To Make

- [ ] Hyperparameter tuning — Optuna vs GridSearchCV?
- [ ] Whether to add external data sources (weather API, events)
- [ ] Deployment approach — FastAPI? Streamlit demo?
- [ ] How to handle seasonal model drift over time
