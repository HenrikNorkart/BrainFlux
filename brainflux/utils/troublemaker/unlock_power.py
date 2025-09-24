# %%
import numpy as np


from brainflux.utils.troublemaker import (
    get_trouble_makers_patient_ids,
    filter_out_trouble_makers,
)


def unlock_power(
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

    data_filtered, labels_filtered = filter_out_trouble_makers(
        data,
        labels,
        keep_percentile=keep_percentile,
        true_positive_label=true_positive_label,
    )

    max_filtered_value = np.max(data_filtered[labels_filtered != true_positive_label])
    res = {}

    for tm, tm_patient_id in zip(data[tm_indices], tm_patient_ids):

        target_data_points = data[labels == true_positive_label]

        count = np.sum(
            (target_data_points > max_filtered_value) & (target_data_points < tm)
        )

        percentile = count / len(target_data_points)

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

    data = res.distribution[:, 0, 0]

    unlock_powers = unlock_power(
        data,
        res.labels,
        patient_ids=res.patient_ids,
        true_positive_label=0,
        keep_percentile=95,
    )

    for patient_id, unlock_power in unlock_powers.items():
        print(f"Patient ID: {patient_id}, Unlock Power: {unlock_power*100:.2f}%")


# %%
