# %%
import numpy as np


from brainflux.utils.troublemaker import (
    get_trouble_makers_patient_ids,
)


def intruder_power(
    data: np.ndarray,
    labels: np.ndarray,
    patient_ids: list[str],
    keep_percentile: int = 95,
    true_positive_label: int = 1,
) -> dict[str, float]:

    tm_patient_ids, tm_indices = get_trouble_makers_patient_ids(
        data,
        labels,
        patient_ids=patient_ids,
        true_positive_label=true_positive_label,
        keep_percentile=keep_percentile,
    )

    std_filtered_value = np.std(data)

    res = {}

    for tm, tm_patient_id in zip(data[tm_indices], tm_patient_ids):

        target_data_points = data[labels == true_positive_label]
        non_target_data_points = data[labels != true_positive_label]

        count_target = np.sum(
            (target_data_points > tm - std_filtered_value)
            & (target_data_points < tm + std_filtered_value)
        )

        count_non_target = np.sum(
            (non_target_data_points > tm - std_filtered_value)
            & (non_target_data_points < tm + std_filtered_value)
        )

        percentile = (
            count_target / (count_target + count_non_target)
            if (count_target + count_non_target) > 0
            else 0
        )
        res[tm_patient_id] = percentile

    return res


if __name__ == "__main__":

    from brainflux.dataloaders import NumpyLoader
    from brainflux.filters import RangeFilter
    from brainflux.aggregators import FilterAggregator
    from brainflux.dataclasses import suppression_ratio

    data_loader = NumpyLoader(label_file="/workspaces/BrainFlux/test_data/train.csv")

    data_filter = RangeFilter(
        data_source=suppression_ratio, num_ranges=4, num_time_divisions=4
    )

    res = FilterAggregator(loader=data_loader, data_filter=data_filter).aggregate()

    data = res.distribution[:, 2, 2]

    intruder_powers = intruder_power(
        data,
        res.labels,
        patient_ids=res.patient_ids,
        true_positive_label=0,
        keep_percentile=95,
    )

    # for patient_id, intruder_power in intruder_powers.items():
    #     print(f"Patient ID: {patient_id}, Intruder Power: {intruder_power*100:.2f}%")


# %%
