import numpy as np

def predict_proba(row):
    # Generated from Scikit-Learn DecisionTree model
    if row['pain_score_mean_x_gcs_score'] <= 0.42668621242046356:
        if row['sedation_score_mean'] <= 1.476269543170929:
            if row['fio2_mean'] <= 51.02120018005371:
                if row['spo2_mean'] <= 96.79093933105469:
                    return 0.18867924528301888
                else:
                    return 0.5147058823529411
            else:
                if row['temperature_trend'] <= -0.0016196846263522586:
                    return 0.25
                else:
                    return 0.04294478527607362
        else:
            if row['gcs_score'] <= 5.5:
                if row['age'] <= 26.5:
                    return 0.17647058823529413
                else:
                    return 0.025
            else:
                if row['temperature_trend'] <= 0.0002640618185978383:
                    return 0.06521739130434782
                else:
                    return 0.3333333333333333
    else:
        if row['fio2_mean'] <= 72.68515396118164:
            if row['temperature_trend'] <= -1.375:
                return 0.0
            else:
                if row['sedation_score_mean'] <= 1.7571428418159485:
                    return 0.9308943089430894
                else:
                    return 0.0
        else:
            if row['age'] <= 54.5:
                return 0.0
            else:
                if row['peep_mean'] <= 9.224929332733154:
                    return 1.0
                else:
                    return 0.0