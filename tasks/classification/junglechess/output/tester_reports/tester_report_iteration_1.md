**Feature Evaluation Report – JungleChess Dataset**

**1. Baseline Model (all available features)**  
- **Algorithm:** RandomForest (n_estimators = 50, n_jobs = 5)  
- **Accuracy:** ≈ 0.605  
- **Feature Importance (gain):**  

| Feature | Importance |
|---|---|
| strength_difference | 0.274 |
| strength_diff_times_distance | 0.274 |
| manhattan_distance | 0.200 |
| rank_offset | 0.088 |
| file_offset | 0.083 |
| white_stronger_flag | 0.081 |
| mouse_vs_elephant_white | 0.0 |
| mouse_vs_elephant_black | 0.0 |

**2. Correlation & Redundancy Analysis**  
- **Strong correlations:**  
  - `strength_difference` ↔ `strength_diff_times_distance` ≈ 0.90  
  - `strength_difference` ↔ `white_stronger_flag` ≈ 0.78  
  - `manhattan_distance` ↔ `rank_offset` ≈ 0.78  
  - `manhattan_distance` ↔ `file_offset` ≈ 0.60  

- **Zero‑importance features:** `mouse_vs_elephant_white`, `mouse_vs_elephant_black`.

**3. Pruning Decision**  
- Removed the following attributes (no predictive contribution and/or high redundancy):  
  - `strength_diff_times_distance`  
  - `mouse_vs_elephant_white`  
  - `mouse_vs_elephant_black`

**4. Post‑pruning Model (5 retained features)**  
- **Algorithm:** RandomForest (n_estimators = 200, n_jobs = 5)  
- **Accuracy:** ≈ 0.603 (virtually unchanged)  
- **Feature Importance:**  

| Feature | Importance |
|---|---|
| strength_difference | 0.457 |
| manhattan_distance | 0.252 |
| white_stronger_flag | 0.116 |
| rank_offset | 0.097 |
| file_offset | 0.080 |

**5. Conclusions**  

- **Predictive Power:** The original feature set yields modest predictive performance (≈ 60 % accuracy).  
- **Key Predictors:** `strength_difference` and `manhattan_distance` dominate importance; they capture the core relational information between the two pieces.  
- **Redundancy:** `strength_diff_times_distance` provides almost identical information to `strength_difference` and can be safely omitted.  
- **Useless Features:** Both `mouse_vs_elephant_*` attributes have zero importance and add no value.  
- **Impact of Pruning:** Removing the three identified features does **not** degrade model performance, confirming they are redundant or irrelevant. The reduced set (5 features) is more parsimonious and easier to interpret.  

**Next Steps for the Team**  
- Keep the pruned attribute list (`strength_diff_times_distance`, `mouse_vs_elephant_white`, `mouse_vs_elephant_black`) removed from further analyses.  
- Focus future hypothesis generation on the retained high‑importance attributes, especially exploring non‑linear interactions between `strength_difference` and `manhattan_distance`.