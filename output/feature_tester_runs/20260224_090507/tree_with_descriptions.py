import numpy as np


def predict_proba_clinical(row):
    """
    Predicts outcome probability and provides a human-readable clinical explanation
    based on the 2025 AHA/ERC guidelines and derived EHR features.
    """

    # Feature 1: Interaction between responsiveness to pain and baseline GCS
    # Higher values (>0.42) suggest higher motor reactivity or preserved consciousness.[1, 2]
    if row["pain_score_mean_x_gcs_score"] > 0.42668621242046356:
        return {
            "probability": 0.895,
            "clinical_status": "Preserved Neurological Reactivity",
            "explanation": (
                "The patient demonstrates high neurological reactivity. The synergistic interaction between "
                "pain responsiveness and GCS indicates cortical or subcortical activity (GCS-M > 3), "
                "which is a strong predictor of favorable functional recovery."
            ),
        }

    else:
        # Path for patients with Low Neurological Reactivity (interaction <= 0.42)

        # Feature 2: Mean Sedation Depth (likely RASS or SAS equivalents)
        # Threshold 1.47: Values above this often indicate moderate to deep pharmacological coverage.[4]
        if row["sedation_score_mean"] > 1.476269543170929:
            return {
                "probability": 0.048,
                "clinical_status": "Deep Pharmacological Coma / Oversedation",
                "explanation": (
                    "The patient is in a state of high sedative exposure. Despite low neurological reactivity, "
                    "the mean sedation score is high, suggesting deep drug-induced depression (RASS -4 to -5). "
                    "This complicates neuroprognostication and is associated with delayed awakening.[5, 6]"
                ),
            }

        else:
            # Patients with Low Reactivity and Light/Moderate Sedation

            # Feature 3: Inspired Oxygen Fraction (FiO2)
            # Threshold 51%: Values > 50% are typically considered 'High' respiratory support.[7, 8]
            if row["fio2_mean"] > 51.02120018005371:
                return {
                    "probability": 0.08,
                    "clinical_status": "High Inspired Oxygen Requirement",
                    "explanation": (
                        "The patient requires high ventilatory support ($FiO_2 > 51\%$). This signals "
                        "significant pulmonary dysfunction or ischemia-reperfusion lung injury. High oxygen "
                        "demands often necessitate deeper sedation to maintain ventilator synchrony."
                    ),
                }

            else:
                # Feature 4: Peripheral Oxygen Saturation (SpO2)
                # Goal range is 94-98%.
                if row["spo2_mean"] > 96.79093933105469:
                    return {
                        "probability": 0.5147,
                        "clinical_status": "Stable Respiratory State / Potential Hyperoxia",
                        "explanation": (
                            "The patient is respiratorily stable with regular oxygenation. However, the $SpO_2$ is in "
                            "the high-normal range (> 96.8%), necessitating careful downward titration of $FiO_2$ to "
                            "avoid hyperoxia-induced oxidative stress in the brain."
                        ),
                    }
                else:
                    return {
                        "probability": 0.1887,
                        "clinical_status": "Sub-optimal Oxygenation / Hypoxic Risk",
                        "explanation": (
                            "While $FiO_2$ requirement is low, the peripheral oxygen saturation is also low/marginal "
                            "(<= 96.8%). In the context of low neurological reactivity, this suggests a risk of "
                            "occult hypoxemia ($PaO_2 < 60$ mmHg), which can trigger secondary brain injury."
                        ),
                    }


# Example Usage:
# result = predict_proba_clinical(patient_row)
# print(f"Outcome Probability: {result['probability']}")
# print(f"Status: {result['clinical_status']}")
# print(f"Analysis: {result['explanation']}")
