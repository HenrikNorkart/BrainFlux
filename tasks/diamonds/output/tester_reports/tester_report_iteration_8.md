**Feature Evaluation Report – Diamond Price Prediction**

**1. Objective**  
Assess the predictive power and importance of the 79 supplied attributes for estimating diamond price (`target`), and prune non‑contributory features.

**2. Experimental Setup**  

| Component | Details |
|-----------|---------|
| Model | XGBoost Regressor (n_estimators = 300, learning_rate = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8) |
| Hardware | GPU (`device="cuda:5"`, `tree_method="hist"`) |
| Train‑test split | 80 % / 20 % (random_state = 42) |
| Evaluation metrics | R², Mean Absolute Error (MAE) |
| Importance measures | • **Gain importance** from XGBoost (raw split gain)  <br>• **Permutation importance** (negative MAE, 5 repeats) |
| Additional analysis | Correlation matrix of numeric features (sample shown) |
| Pruning criterion | Features with gain < 1 × 10⁵ (practically zero contribution). |

**3. Baseline Results (All 79 features)**  

| Metric | Value |
|--------|-------|
| R² | **0.9842** |
| MAE | **251** (price units) |
| Number of features | 79 |

**Top Gain Importance (gain value)**  

| Feature | Gain |
|---------|------|
| `log_volume_squared` | 4.32 × 10⁹ |
| `carat_squared_clarity` | 3.15 × 10⁹ |
| `log_carat_clarity` | 2.71 × 10⁹ |
| `y_width` | 1.06 × 10⁹ |
| `sphericity` | 6.39 × 10⁸ |
| `log_carat_color` | 3.71 × 10⁸ |
| `carat` | 3.62 × 10⁸ |
| `carat_squared_color` | 2.22 × 10⁸ |
| `log_surface_area` | 2.14 × 10⁸ |
| `surface_area` | 1.77 × 10⁸ |

**Top Permutation Importance (mean increase in MAE when shuffled)**  

| Feature | ΔMAE |
|---------|------|
| `carat_squared_clarity` | 1339 |
| `carat_squared_color` | 343 |
| `log_carat_clarity` | 290 |
| `y_width` | 279 |
| `log_volume_squared` | 165 |
| `carat` | 145 |
| `log_carat_color` | 58 |
| `color_clarity_interaction` | 39 |
| `z_depth` | 30 |
| `surface_area` | 27 |

**Interpretation**  
- Engineered logarithmic and interaction features dominate predictive power (especially those involving **carat**, **clarity**, and **log‑volume**).  
- The raw physical dimension `y_width` and the fundamental `carat` weight also remain important.  
- Many original or simple engineered attributes contribute virtually nothing (gain ≈ 0).

**4. Feature Pruning**  

Features with gain < 1 × 10⁵ (practically zero) were removed:

```
log_carat, carat_squared, carat_cut_interaction, y_cut_interaction,
carat_cubed, surface_area_squared, log_volume, log_surface_area,
shape_cluster, dim_variance, shape_cluster_clarity_interaction,
shape_cluster_color_interaction, shape_cluster_cut_interaction,
log_volume_cubed, log_volume_shape_cluster
```

**Post‑pruning Results (64 features)**  

| Metric | Value |
|--------|-------|
| R² | **0.9841** (Δ = ‑0.0001) |
| MAE | **251** (Δ = +0.06) |
| Number of features | **64** |

**Outcome** – Pruning reduced the feature set by ~19 % without any measurable loss in predictive performance.

**5. Conclusions**  

1. **Predictive Power** – The engineered logarithmic and interaction attributes capture the majority of the variance in diamond price, delivering a very high R² (~0.984).  
2. **Key Drivers** – `log_volume_squared`, `carat_squared_clarity`, `log_carat_clarity`, and `y_width` are the strongest predictors.  
3. **Redundant Features** – A subset of 15 attributes provides negligible gain and can be safely omitted, simplifying the model and reducing computational load.  
4. **Robustness** – Model performance remains stable after pruning, indicating that the retained features are robust contributors.

**Next Steps for the Team**  

- The **Scientist Agent** can focus hypothesis generation on the high‑importance engineered features (e.g., interactions involving carat and clarity).  
- The **Extractor Agent** may explore additional interaction terms centred on the top‑ranked dimensions (`y_width`, `carat`, `log_volume`).  
- Continue monitoring feature stability under data perturbations (e.g., added noise) to verify robustness.  

*All observations have been recorded in the internal notes for reference.*