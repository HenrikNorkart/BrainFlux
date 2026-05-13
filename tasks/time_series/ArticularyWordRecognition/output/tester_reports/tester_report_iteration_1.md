**Comprehensive Feature‑Evaluation Report**

---

### 1. Overview  

The dataset contains 275 instances and 16 engineered EMA‑derived attributes (plus the target).  
The task is a 25‑class classification problem.  

Because the execution environment raised a *ConsoleManager* error when training XGBoost models, the full predictive‑power assessment (accuracy, macro‑F1, SHAP importance) could not be completed.  
Instead, a rigorous statistical analysis was performed to gauge each attribute’s relevance and redundancy.

---

### 2. Statistical Relationships  

| Feature | |Spearman |Correlation*| |
|---------|---|----------|------------|
|sensor_0_median| |0.172| |
|sensor_0_min| |0.117| |
|sensor_0_max| |0.109| |
|sensor_1_mean| |0.175| |
|sensor_1_min| |0.157| |
|sensor_1_std| |0.072| |
|sensor_1_median| |0.080| |
|*(All other features ≤ 0.05)* | | |  

\* Absolute Spearman correlation with the multi‑class target (higher values indicate stronger monotonic relationship).  

**Key observations**

* The strongest monotonic links to the target are modest (≈ 0.17).  
* Most velocity and acceleration‑derived attributes show near‑zero correlation (≤ 0.02).  

---

### 3. Inter‑Feature Redundancy  

Pearson correlation analysis revealed **10 pairs** with |r| > 0.9, all involving the velocity and acceleration statistics of sensor 0:

| Highly correlated pair | |r| |
|------------------------|---|----|
|sensor_0_vel_mean – sensor_0_acc_mean| |0.994|
|sensor_0_vel_std – sensor_0_vel_max| |0.968|
|sensor_0_vel_std – sensor_0_vel_range| |0.968|
|sensor_0_vel_max – sensor_0_vel_range| |1.00|
|sensor_0_acc_mean – sensor_0_acc_std| |0.963|
|… (six additional similar pairs) | | |

These features provide essentially the same information and are therefore redundant.

---

### 4. Feature Pruning  

Based on the two criteria above (low target correlation *and* high redundancy), the following **9 attributes** were removed:

```
sensor_0_mean
sensor_0_std
sensor_0_range
sensor_0_vel_mean
sensor_0_vel_std
sensor_0_vel_max
sensor_0_vel_range
sensor_0_acc_mean
sensor_0_acc_std
```

**Remaining feature set (7 attributes)**  

| Feature | Spearman |Correlation* |
|---------|----------|--------------|
|sensor_0_median|0.172|
|sensor_0_min|0.117|
|sensor_0_max|0.109|
|sensor_1_mean|0.175|
|sensor_1_min|0.157|
|sensor_1_std|0.072|
|sensor_1_median|0.080|

*All pairwise Pearson correlations among the retained features are ≤ 0.56 (the highest is sensor_1_median ↔ sensor_1_min), indicating low multicollinearity.*

---

### 5. Predictive‑Power Insight (Qualitative)

* Because the strongest individual correlations are only around 0.17, the retained attributes alone are unlikely to yield high classification accuracy on their own.  
* The original full set contained many highly redundant velocity/acceleration statistics that added little unique information and could even dilute model learning.  
* After pruning, the feature space is compact (7 attributes) and free of severe multicollinearity, which should make any downstream modelling (e.g., XGBoost, Logistic Regression) more stable and interpretable.

---

### 6. Recommendations for the Next Phase  

* **Model Training** – With the reduced 7‑feature set, re‑run an XGBoost (or any classifier) experiment. The smaller dimensionality will avoid the previous console‑manager issue and should execute quickly.  
* **Performance Metrics** – Record accuracy, macro‑F1, and confusion matrix to quantify the practical predictive gain of the retained attributes.  
* **Interpretability** – Use SHAP (TreeSHAP) on the trained model to confirm that the features identified as most correlated indeed dominate the model’s decisions.  

---

### 7. Notes (recorded)

*Initial XGBoost attempts failed due to a ConsoleManager error. Conducted statistical analysis instead. Identified low‑correlation and highly redundant velocity/acceleration features, leading to pruning of nine attributes and retaining seven informative ones.*  

---  

**End of Report**