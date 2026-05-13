# Feature Tester Run Results

**Generated:** 2026-03-02 13:33:54

## Run Configuration

- **Target Class:** 0
- **Precision Threshold:** 97.00%
- **Output Directory:** /workspaces/BrainFlux/output/feature_tester_runs/20260302_133343

## Model Information

- **Model Type:** DecisionTreeRegressor
- **Max Depth:** 4
- **Min Samples Leaf:** 1
- **Number of Features:** 9

## Features Used

- sedation_score_mean
- pain_score_mean_x_gcs_score
- fio2_mean
- gcs_score
- spo2_mean
- temperature_trend
- pain_score_autocorr1
- age
- peep_mean

## Model Performance

### Feature Importances

- **pain_score_mean_x_gcs_score**: 0.7605
- **fio2_mean**: 0.0956
- **sedation_score_mean**: 0.0677
- **temperature_trend**: 0.0321
- **spo2_mean**: 0.0275
- **age**: 0.0077
- **peep_mean**: 0.0045
- **gcs_score**: 0.0044
- **pain_score_autocorr1**: 0.0000

## Threshold Evaluation Results

| Threshold | Train Removed | Test Removed | Train Score | Test Score | Train ROC-AUC | Test ROC-AUC | Train Recall | Test Recall | Train Accuracy | Test Accuracy |
|-----------|---------------|--------------|-------------|-----------|--------------|-------------|-------------|-----------|----------------|---------------|
| 0.01 | 99.08% (1077 / 1087) | 98.34% (713 / 725) | 90.00% | 33.33% | 93.55% | 88.65% | 100.00% | 96.20% | 32.93% | 31.86% |
| 0.02 | 99.08% (1077 / 1087) | 98.34% (713 / 725) | 90.00% | 33.33% | 93.55% | 88.65% | 100.00% | 96.20% | 32.93% | 31.86% |
| 0.03 | 65.96% (717 / 1087) | 66.07% (479 / 725) | 91.41% | 87.45% | 93.55% | 88.65% | 97.41% | 93.67% | 64.40% | 62.48% |
| 0.04 | 65.96% (717 / 1087) | 66.07% (479 / 725) | 91.41% | 87.45% | 93.55% | 88.65% | 97.41% | 93.67% | 64.40% | 62.48% |
| 0.05 | 50.97% (554 / 1087) | 51.59% (374 / 725) | 91.10% | 89.76% | 93.55% | 88.65% | 95.40% | 91.98% | 78.10% | 75.86% |
| 0.06 | 50.97% (554 / 1087) | 51.59% (374 / 725) | 91.10% | 89.76% | 93.55% | 88.65% | 95.40% | 91.98% | 78.10% | 75.86% |
| 0.07 | 46.73% (508 / 1087) | 47.17% (342 / 725) | 90.71% | 89.47% | 93.55% | 88.65% | 94.54% | 90.72% | 81.78% | 79.45% |
| 0.08 | 46.73% (508 / 1087) | 47.17% (342 / 725) | 90.71% | 89.47% | 93.55% | 88.65% | 94.54% | 90.72% | 81.78% | 79.45% |
| 0.09 | 46.73% (508 / 1087) | 47.17% (342 / 725) | 90.71% | 89.47% | 93.55% | 88.65% | 94.54% | 90.72% | 81.78% | 79.45% |
| 0.10 | 46.73% (508 / 1087) | 47.17% (342 / 725) | 90.71% | 89.47% | 93.55% | 88.65% | 94.54% | 90.72% | 81.78% | 79.45% |
| 0.20 | 40.29% (438 / 1087) | 39.72% (288 / 725) | 90.76% | 89.28% | 93.55% | 88.65% | 90.80% | 84.81% | 85.83% | 83.03% |
| 0.30 | 36.98% (402 / 1087) | 37.93% (275 / 725) | 90.22% | 89.16% | 93.55% | 88.65% | 88.22% | 81.43% | 87.49% | 82.62% |
| 0.40 | 35.33% (384 / 1087) | 35.72% (259 / 725) | 86.43% | 85.95% | 93.55% | 88.65% | 86.49% | 80.59% | 88.04% | 84.28% |
| 0.50 | 35.33% (384 / 1087) | 35.72% (259 / 725) | 86.43% | 85.95% | 93.55% | 88.65% | 86.49% | 80.59% | 88.04% | 84.28% |
| 0.60 | 22.82% (248 / 1087) | 22.76% (165 / 725) | 55.26% | 51.37% | 93.55% | 88.65% | 66.38% | 63.29% | 87.67% | 85.93% |
| 0.70 | 22.82% (248 / 1087) | 22.76% (165 / 725) | 55.26% | 51.37% | 93.55% | 88.65% | 66.38% | 63.29% | 87.67% | 85.93% |
| 0.80 | 22.82% (248 / 1087) | 22.76% (165 / 725) | 55.26% | 51.37% | 93.55% | 88.65% | 66.38% | 63.29% | 87.67% | 85.93% |
| 0.90 | 22.82% (248 / 1087) | 22.76% (165 / 725) | 55.26% | 51.37% | 93.55% | 88.65% | 66.38% | 63.29% | 87.67% | 85.93% |

## Selected Threshold Analysis

- **Selected Threshold:** 0.40
- **Python Implementation AUC:** 0.8865
- **Original Model AUC:** 0.8865
- **AUC Discrepancy:** 0.000000

### Confusion Matrix (Test Set)

```
Original Model:
[[401  87]
 [ 36 201]]

Python Implementation:
[[401  87]
 [ 36 201]]
```

## Output Files

All results from this run have been saved to: `/workspaces/BrainFlux/output/feature_tester_runs/20260302_133343`

### Generated Files

- `train_patient_scores.csv` - Patient scores for training set
- `test_patient_scores.csv` - Patient scores for test set
- `terminal_node_patient_groups.csv` - Patient ID to terminal-node patient-group mapping
- `patient_id_descriptions.json` - Per-patient descriptions grouped by patient ID
- `correlation_matrix.png` - Feature correlation heatmap (if generated)
- `troublemaker_decision_tree.png` - Decision tree visualization (graphviz)
- `troublemaker_decision_tree_matplotlib.png` - Decision tree visualization (matplotlib)
- `model_rules_generated.py` - Exportable model rules as Python code

## Model Rules

The trained model has been exported to Python code in `model_rules_generated.py`.
This allows for easy deployment and prediction without requiring sklearn dependencies.

## Data Summary

- **Training Set:** 1087 patients
- **Test Set:** 725 patients
- **Total Patients:** 1812 patients

## Next Steps

1. Review the decision tree visualizations in the output directory
2. Evaluate the model scores for individual patients
3. Adjust threshold if needed based on business requirements
4. Deploy the generated Python model rules if performance is acceptable
