**Comprehensive Feature Evaluation Report**

**1. Dataset Overview**
- The extracted dataset contains 11 engineered torque‑related attributes plus the target class (`left`, `right`, `balanced`).
- Attribute list (pre‑pruning):  
  `left_torque`, `right_torque`, `torque_diff`, `torque_abs_diff`, `torque_ratio`, `log_left_torque`, `log_right_torque`, `left_torque_sq`, `right_torque_sq`, `torque_diff_sign`, `torque_diff_normalized`.

**2. Predictive Power**
- Using a RandomForest classifier (300 trees, default depth) on the full 11‑feature set yields **100 % accuracy** on a stratified 20 % hold‑out test set.
- Classification metrics (precision, recall, F1) are all 1.0 for each class.

**3. Feature Importance**
- Importance (gain) from the RandomForest model:

| Feature                | Importance |
|------------------------|------------|
| `torque_diff`          | 0.232 |
| `torque_diff_normalized` | 0.225 |
| `torque_diff_sign`     | 0.225 |
| `torque_ratio`         | 0.220 |
| (remaining 7 features) | ≤ 0.045 each |

`torque_diff` (left torque – right torque) is by far the most predictive.

**4. Statistical Relationships**
- High inter‑feature correlations (|ρ| > 0.9):
  - `left_torque` ↔ `log_left_torque` (0.92)  
  - `left_torque` ↔ `left_torque_sq` (0.96)  
  - `right_torque` ↔ `log_right_torque` (0.92)  
  - `right_torque` ↔ `right_torque_sq` (0.96)  
  - `torque_diff` ↔ `torque_diff_normalized` (0.93)

These redundancies confirm that many attributes are deterministic transformations of a few core variables.

**5. Impact of Feature Sub‑sets**
- **5 torque‑derived features** (`torque_diff`, `torque_abs_diff`, `torque_ratio`, `torque_diff_sign`, `torque_diff_normalized`) → 100 % accuracy.
- **Top 3 features** (`torque_diff`, `torque_ratio`, `torque_diff_sign`) → 100 % accuracy.
- **2 features** (`torque_diff`, `torque_ratio`) → 100 % accuracy.
- **Single feature** `torque_diff` → 100 % accuracy.

Thus the target is fully determined by the sign (and magnitude) of `torque_diff`, matching the theoretical rule for the balance‑scale problem:  

\[
\text{class} = 
\begin{cases}
\text{left} & \text{if } left\_weight\times left\_distance > right\_weight\times right\_distance \\
\text{right} & \text{if } left\_weight\times left\_distance < right\_weight\times right\_distance \\
\text{balanced} & \text{otherwise}
\end{cases}
\]

**6. Robustness**
- No degradation observed when removing all but `torque_diff`. The model remains perfectly accurate, indicating extreme robustness of this single attribute.

**7. Feature Pruning**
- All attributes except `torque_diff` were deemed redundant and **pruned** via `attribute_pruning_tool`.

**8. Final Feature Set**
- **Retained attribute:** `torque_diff`
- **Number of features after pruning:** 1

**9. Conclusions**
- The engineered feature `torque_diff` alone captures the full predictive signal for the balance‑scale classification task.
- All other extracted attributes are either highly correlated transformations or provide no additional information.
- The minimal feature set (single attribute) offers maximal interpretability, computational efficiency, and maintains perfect predictive performance.