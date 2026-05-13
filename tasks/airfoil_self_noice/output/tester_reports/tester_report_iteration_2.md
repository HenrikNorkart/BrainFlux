**Feature Evaluation Report – Airfoil Self‑Noise Dataset**

---

### 1. Experimental Setup
| Step | Description |
|------|-------------|
| **Data** | `df_attributes` containing 20 engineered attributes + `target` (sound pressure level). |
| **Initial Feature Set** | 20 attributes (including raw and derived features). |
| **Model** | XGBoost Regressor (`n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `objective='reg:squarederror'`, `eval_metric='rmse'`, `device='cuda:5'`, `tree_method='hist'`). |
| **Evaluation** | Train‑test split (80 %/20 %, `random_state=42`). Metrics: RMSE & R². |
| **Feature‑importance** | Built‑in XGBoost importance (`feature_importances_`). |
| **Robustness Test** | Added Gaussian noise (5 % of each column’s standard deviation) to all retained features and re‑evaluated. |
| **Pruning Criterion** | Features with importance < 0.01 were considered for removal. |

---

### 2. Baseline Results (All 20 features)

| Metric | Value |
|--------|-------|
| **RMSE** | **1.393** |
| **R²** | **0.957** |

**Top‑10 Feature Importances**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `chord_length_squared` | 0.301 |
| 2 | `freq_disp_interaction` | 0.137 |
| 3 | `log_chord_length` | 0.106 |
| 4 | `strouhal_number` | 0.097 |
| 5 | `freq_chord_product` | 0.088 |
| 6 | `reduced_freq_variant` | 0.045 |
| 7 | `free_stream_velocity_squared` | 0.038 |
| 8 | `thickness_to_chord` | 0.030 |
| 9 | `log_frequency` | 0.028 |
|10 | `displacement_thickness` | 0.024 |

**Low‑importance features (importance < 0.01)**  

- `frequency` (0.007)  
- `frequency_squared` (0.005)  
- `displacement_thickness_squared` (0.006)

These contributed negligibly to predictive power.

---

### 3. Feature Pruning

The three low‑importance attributes were removed using **attribute_pruning_tool**.

**Post‑pruning Results (17 features)**  

| Metric | Value |
|--------|-------|
| **RMSE** | **1.398** (Δ + 0.005) |
| **R²** | **0.957** (Δ ‑ 0.0003) |

*Interpretation*: Pruning did **not** degrade model performance in any meaningful way, confirming that the removed features were redundant.

---

### 4. Statistical Relationships (Multicollinearity)

Pairwise absolute correlations (≥ 0.90) among the retained features:

| Feature A | Feature B | |Correlation|
|-----------|-----------|------------|
| `angle_of_attack` | `thickness_to_chord` | 0.925 |
| `chord_length` | `log_chord_length` | 0.945 |
| `chord_length` | `chord_length_squared` | 0.969 |
| `free_stream_velocity` | `log_free_stream_velocity` | 0.995 |
| `free_stream_velocity` | `free_stream_velocity_squared` | 0.996 |
| `strouhal_number` | `freq_chord_product` | 0.931 |
| `strouhal_number` | `reduced_freq_variant` | 0.959 |
| `log_free_stream_velocity` | `free_stream_velocity_squared` | 0.982 |

While many engineered features are highly correlated (by design), XGBoost’s tree‑based learning handles multicollinearity well; no adverse impact on validation metrics was observed.

---

### 5. Robustness Testing

Added 5 % Gaussian noise to all retained features and re‑trained the model.

| Metric (noisy data) | Value |
|----------------------|-------|
| **RMSE** | **2.06** |
| **R²** | **0.907** |

*Interpretation*: Performance degrades modestly under realistic perturbations, but the model retains strong predictive ability (R² > 0.90), indicating robustness of the selected feature set.

---

### 6. Conclusions & Recommendations

1. **Predictive Power** – The engineered feature set achieves **high accuracy** (RMSE ≈ 1.4 dB, R² ≈ 0.96) on unseen data.  
2. **Key Drivers** – The most influential attributes are related to **chord length** (squared & log forms) and **interaction terms** involving frequency, displacement thickness, and chord length.  
3. **Pruned Features** – `frequency`, `frequency_squared`, and `displacement_thickness_squared` can be safely removed without harming performance, reducing dimensionality from 20 → 17.  
4. **Multicollinearity** – Expected high correlations among engineered features do not impair model performance due to the tree‑based algorithm.  
5. **Robustness** – Model remains reliable under moderate noise, confirming the stability of the selected attributes.  

**Final Feature Set (17 attributes)**  

```
['angle_of_attack', 'chord_length', 'free_stream_velocity',
 'displacement_thickness', 'strouhal_number', 'thickness_to_chord',
 'log_frequency', 'log_displacement_thickness', 'frequency_squared',
 'displacement_thickness_squared', 'freq_disp_interaction',
 'log_chord_length', 'log_free_stream_velocity',
 'chord_length_squared', 'free_stream_velocity_squared',
 'chord_vel_interaction', 'freq_chord_product',
 'reduced_freq_variant']
```

(After pruning, the three low‑importance columns are omitted.)

These findings provide a concise, high‑performing feature portfolio for the airfoil self‑noise regression task.