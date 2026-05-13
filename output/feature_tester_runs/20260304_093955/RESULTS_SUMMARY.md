# Feature Tester Run Results

**Generated:** 2026-03-04 09:40:07

## Run Configuration

- **Target Class:** 1
- **Precision Threshold:** 100.00%
- **Output Directory:** /workspaces/BrainFlux/output/feature_tester_runs/20260304_093955

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
| 0.01 | 99.82% (1085 / 1087) | 99.72% (723 / 725) | 50.00% | 0.00% | 93.55% | 88.65% | 100.00% | 99.59% | 68.17% | 67.03% |
| 0.02 | 99.82% (1085 / 1087) | 99.72% (723 / 725) | 50.00% | 0.00% | 93.55% | 88.65% | 100.00% | 99.59% | 68.17% | 67.03% |
| 0.03 | 99.82% (1085 / 1087) | 99.72% (723 / 725) | 50.00% | 0.00% | 93.55% | 88.65% | 100.00% | 99.59% | 68.17% | 67.03% |
| 0.04 | 99.82% (1085 / 1087) | 99.72% (723 / 725) | 50.00% | 0.00% | 93.55% | 88.65% | 100.00% | 99.59% | 68.17% | 67.03% |
| 0.05 | 99.82% (1085 / 1087) | 99.72% (723 / 725) | 50.00% | 0.00% | 93.55% | 88.65% | 100.00% | 99.59% | 68.17% | 67.03% |
| 0.06 | 99.82% (1085 / 1087) | 99.72% (723 / 725) | 50.00% | 0.00% | 93.55% | 88.65% | 100.00% | 99.59% | 68.17% | 67.03% |
| 0.07 | 77.18% (839 / 1087) | 77.24% (560 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 97.70% | 96.93% | 87.67% | 85.93% |
| 0.08 | 77.18% (839 / 1087) | 77.24% (560 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 97.70% | 96.93% | 87.67% | 85.93% |
| 0.09 | 77.18% (839 / 1087) | 77.24% (560 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 97.70% | 96.93% | 87.67% | 85.93% |
| 0.10 | 77.18% (839 / 1087) | 77.24% (560 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 97.70% | 96.93% | 87.67% | 85.93% |
| 0.20 | 77.18% (839 / 1087) | 77.24% (560 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 97.70% | 96.93% | 87.67% | 85.93% |
| 0.30 | 77.18% (839 / 1087) | 77.24% (560 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 97.70% | 96.93% | 87.67% | 85.93% |
| 0.40 | 77.18% (839 / 1087) | 77.24% (560 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 97.70% | 96.93% | 87.67% | 85.93% |
| 0.50 | 64.67% (703 / 1087) | 64.28% (466 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 88.77% | 86.07% | 88.04% | 84.28% |
| 0.60 | 64.67% (703 / 1087) | 64.28% (466 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 88.77% | 86.07% | 88.04% | 84.28% |
| 0.70 | 63.02% (685 / 1087) | 62.07% (450 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 87.14% | 83.20% | 87.49% | 82.62% |
| 0.80 | 59.71% (649 / 1087) | 60.28% (437 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 83.49% | 82.17% | 85.83% | 83.03% |
| 0.90 | 53.27% (579 / 1087) | 52.83% (383 / 725) | 0.00% | 0.00% | 93.55% | 88.65% | 75.78% | 73.98% | 81.78% | 79.45% |

## Selected Threshold Analysis

- **Selected Threshold:** 0.40
- **Python Implementation AUC:** 0.1135
- **Original Model AUC:** 0.8865
- **AUC Discrepancy:** 0.772904

### Confusion Matrix (Test Set)

```
Original Model:
[[150  87]
 [ 15 473]]

Python Implementation:
[[ 36 201]
 [401  87]]
```

## Output Files

All results from this run have been saved to: `/workspaces/BrainFlux/output/feature_tester_runs/20260304_093955`

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
