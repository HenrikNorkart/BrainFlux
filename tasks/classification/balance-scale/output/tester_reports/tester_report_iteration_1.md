**Feature Evaluation Report – Balance‑Scale Dataset**

---

### 1. Overall Predictive Power  
- **Model:** XGBoost (multi‑class, GPU `cuda:5`, `tree_method='hist'`)  
- **Evaluation:** 5‑fold stratified cross‑validation  
- **Result:** **100 % mean accuracy** (all folds achieved 1.0).  
- **Interpretation:** The supplied feature set is fully sufficient to deterministically separate the three classes.

---

### 2. Feature Importance (XGBoost “gain”)  

| Rank | Feature | Gain (relative) |
|------|---------|-----------------|
| 1 | **torque_ratio** | 31.92 |
| 2 | **torque_diff**  | 28.22 |
| 3 | **torque_abs_diff** | 14.74 |
| 4‑9 | *left_torque, right_torque, log_left_torque, log_right_torque, left_torque_sq, right_torque_sq* | **0.0** (not used in any split) |

*Only the three torque‑derived features contributed to model decisions.*

---

### 3. Statistical Relationships & Redundancy  

- **Correlation matrix** (excerpt) shows very high collinearity among the pruned features:  
  - `left_torque` ↔ `log_left_torque` ≈ 0.92  
  - `left_torque` ↔ `left_torque_sq` ≈ 0.96  
  - Similar patterns for the right‑side counterparts.  

- **Implication:** The high‑correlation group provides no additional information beyond the three “difference/ratio” features that already capture the physics of the balance scale.

---

### 4. Robustness Checks  

- After **pruning** the six redundant attributes, the model still achieved **100 % accuracy** (identical to the unpruned version).  
- This confirms that the removed features are **non‑essential** and do not affect predictive performance.

---

### 5. Pruning Action  

The following attributes were removed using the `attribute_pruning_tool`:

```
left_torque
right_torque
log_left_torque
log_right_torque
left_torque_sq
right_torque_sq
```

The remaining feature set is:

```
torque_ratio
torque_diff
torque_abs_diff
target
```

---

### 6. Key Take‑aways  

1. **Predictive Power:** The engineered torque‑based features perfectly predict the balance‑scale class.  
2. **Feature Importance:** `torque_ratio`, `torque_diff`, and `torque_abs_diff` are the only useful attributes.  
3. **Redundancy:** Original raw torque values, their logs, and squared terms are redundant and contribute zero importance.  
4. **Pruning Outcome:** Removing the redundant six attributes does **not** degrade model performance, yielding a compact, interpretable feature set.

---

**Conclusion:**  
The dataset’s discriminative information is fully captured by the three torque‑derived features. Maintaining only these three attributes provides a minimal, high‑performing model without loss of accuracy.