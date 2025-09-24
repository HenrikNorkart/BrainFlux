import numpy as np


def filter_discriminative_power(
    filter_scores: np.ndarray, labels: np.ndarray, *, non_zero: bool = False
) -> float:

    data_non_survivor = filter_scores[labels == 0]
    data_survivor = filter_scores[labels == 1]

    def order_data(data):
        filter_scores = np.linspace(0, 1, 500)
        percentages = np.array(
            [
                (data >= score).sum() / len(data) if len(data) > 0 else 0
                for score in filter_scores
            ]
        )
        return filter_scores, percentages

    data_non_survivor_ordered, percentages_non_survivor = order_data(data_non_survivor)
    data_survivor_ordered, percentages_survivor = order_data(data_survivor)

    area_non_survivor = np.trapz(percentages_non_survivor, data_non_survivor_ordered)
    area_survivor = np.trapz(percentages_survivor, data_survivor_ordered)
    fdp = (area_survivor - area_non_survivor) / max(area_survivor, area_non_survivor)

    if non_zero:
        fdp = abs(fdp)

    return fdp
