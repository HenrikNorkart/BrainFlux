import numpy as np


def evaluate_suppression_ratio_reliability(row):
    """
    Evaluates whether a negative post-cardiac arrest outcome can be determined
    solely by a high suppression ratio based on interacting EHR features.

    Interpretive Scale:
    - Score ~1.0: Low Certainty in suppression ratio (External confounders present).
    - Score ~0.0: High Certainty in suppression ratio (Determined Negative Outcome).
    """

    # 1. PRIMARY BRANCH: Neurological Reactivity (Pain x GCS Interaction)
    # Clinical reactivity (GCS Motor > 3) is a primary external factor that
    # contradicts the finality of a high suppression ratio.[1, 2]
    if row["pain_score_mean_x_gcs_score"] > 0.42668621242046356:
        return {
            "patient_group": 1,
            "ambiguity_score": 0.895,
            "patient_mass": 23.7,
            "interpretation": "Low Certainty in Standalone Suppression Ratio",
            "clinical_status": "Preserved Neurological Reactivity",
            "numeric_logic": f"High Interaction ($Score > 0.43$ out of $\approx 90$)",
            "explanation": (
                "The patient demonstrates preserved reactivity to stimuli. This 'fast' response "
                "is a major external factor that conflicts with a high suppression ratio. "
                "Because the patient shows signs of cortical/subcortical activity (likely "
                "GCS Motor score $> 3$ out of $6$), a negative outcome cannot be determined "
                "solely based on the EEG reading.[3, 4]"
            ),
        }

    else:
        # 2. SECONDARY BRANCH: Low Reactivity Pathway

        # 2a. Pharmacological Interaction (Mean Sedation)
        # Deep sedation (SAS 1-2 or RASS -4/-5) is congruent with high suppression.[5, 6]
        if row["sedation_score_mean"] > 1.476269543170929:
            return {
                "patient_group": 2,
                "ambiguity_score": 0.048,
                "patient_mass": 40.6,
                "interpretation": "High Certainty of Determined Negative Outcome",
                "clinical_status": "Deep Pharmacological Suppression",
                "numeric_logic": f"High Sedation Mean ($Score > 1.48$ out of $7$)",
                "explanation": (
                    "The patient is in a state of 'slow' reactivity and deep pharmacological "
                    "suppression. This high level of sedative exposure is congruent with a "
                    "high suppression ratio. In this profoundly depressed state, the model "
                    "finds high certainty that the outcome is negative.[7, 8]"
                ),
            }

        else:
            # 2b. Systemic Stress Interaction (FiO2 Mean)
            # High FiO2 requirements (> 50%) signal severe PCAS systemic injury.[9, 10]
            if row["fio2_mean"] > 51.02120018005371:
                return {
                    "patient_group": 3,
                    "ambiguity_score": 0.08,
                    "patient_mass": 18.3,
                    "interpretation": "High Certainty of Determined Negative Outcome",
                    "clinical_status": "High Respiratory Support / Metabolic Stress",
                    "numeric_logic": f"High $FiO_2$ ($Fraction > 51.02\%$)",
                    "explanation": (
                        "The patient requires 'heavy' ventilatory support with high oxygen "
                        "fractions ($FiO_2 > 51\%$). This signals severe systemic ischemia-reperfusion "
                        "injury and metabolic stress. This systemic failure aligns with a high "
                        "suppression ratio, allowing for a clear determination of a negative outcome."
                    ),
                }

            else:
                # 2c. Oxygenation Stability (SpO2 Mean)
                # AHA/ERC target 94–98%. Lower values increase hypoxic brain injury risk.[9, 13]
                if row["spo2_mean"] <= 96.79093933105469:
                    return {
                        "patient_group": 4,
                        "ambiguity_score": 0.189,
                        "patient_mass": 4.9,
                        "interpretation": "High Certainty of Determined Negative Outcome",
                        "clinical_status": "Marginal Oxygenation / Hypoxic Risk",
                        "numeric_logic": f"Low/Marginal $SpO_2$ ($Saturation \leq 96.8\%$)",
                        "explanation": (
                            "Neurological reactivity is low and peripheral oxygenation is marginal/irregular "
                            "($SpO_2 \leq 96.8\%$). These sub-optimal conditions confirm that the suppression "
                            "ratio reflects genuine pathological brain injury rather than external "
                            "physiological fluctuations.[1, 14]"
                        ),
                    }
                else:
                    # 2d. Stable Physiology (High SpO2 with low support)
                    return {
                        "patient_group": 5,
                        "ambiguity_score": 0.5147,
                        "patient_mass": 12.5,
                        "interpretation": "Moderate Certainty / Ambiguous",
                        "clinical_status": "Stable Respiratory State / Balanced Oxygenation",
                        "numeric_logic": f"Regular $SpO_2$ ($Saturation > 96.8\%$)",
                        "explanation": (
                            "The patient maintains a 'regular' oxygenation profile ($SpO_2 > 96.8\%$) "
                            "with low inspired oxygen requirements and only moderate sedation. This "
                            "physiological stability is an external factor that interacts with the "
                            "EEG readings, making it difficult to determine a negative outcome "
                            "based on the suppression ratio alone.[15, 9]"
                        ),
                    }
