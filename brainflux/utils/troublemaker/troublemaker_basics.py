from typing import Iterable
from functools import reduce
from copy import deepcopy

import numpy as np


def get_trouble_makers_indices(
    data: np.ndarray,
    labels: np.ndarray,
    keep_percentile: float = 95,
    true_positive_label: int = 1,
    descending_sort: bool = True,
) -> np.ndarray:
    """Get the indices of trouble makers in the data.

    Args:
        data (np.ndarray): The input data array. Shape (num_samples, ...).
        labels (np.ndarray): The labels corresponding to the data. Shape (num_samples,).
        keep_percentile (float, optional): The percentile of non-target samples to keep. Defaults to 95.
        true_positive_label (int, optional): The label considered as the true positive. Defaults to 1.

    Returns:
        np.ndarray: The indices of the trouble makers per data channel. Shape (num_trouble_makers, ...).
    """

    assert len(data) == len(labels), "Data and labels must have the same length."
    # Find indices where labels != true_positive_label
    non_target_indices = np.where(labels != true_positive_label)[0]

    num_trouble_makers_to_find = min(
        int(len(non_target_indices) * ((100.0 - keep_percentile) / 100.0)),
        len(non_target_indices),
    )

    if data.ndim == 1:
        num_channels = 1
    else:
        num_channels = reduce(lambda x, y: x * y, [d for d in data.shape[1:] if d > 0])

    results = np.ndarray(
        (num_trouble_makers_to_find, num_channels),
        dtype=int,
    )

    if len(non_target_indices) > 0:
        # Get scores for non-target labels
        non_target_scores = data[non_target_indices]
        non_target_scores = non_target_scores.reshape(non_target_scores.shape[0], -1)

        for channel in range(num_channels):
            channel_scores = non_target_scores[:, channel]
            non_target_scores_and_indices = list(
                zip(channel_scores, non_target_indices)
            )
            non_target_scores_and_indices.sort(
                reverse=descending_sort, key=lambda x: x[0]
            )
            sorted_non_target_scores, sorted_non_target_indices = zip(
                *non_target_scores_and_indices
            )

            sorted_non_target_scores = sorted_non_target_scores[
                :num_trouble_makers_to_find
            ]
            sorted_non_target_indices = sorted_non_target_indices[
                :num_trouble_makers_to_find
            ]

            results[:, channel] = sorted_non_target_indices

    if data.ndim == 1:
        data_shape = [num_trouble_makers_to_find, 1]

    else:
        data_shape = list(data.shape)
        data_shape[0] = num_trouble_makers_to_find

    return results.reshape(*data_shape)


def get_trouble_makers_patient_ids(
    data: np.ndarray,
    labels: np.ndarray,
    patient_ids: list[str],
    channel: int | Iterable[int] | None = None,
    keep_percentile: int = 95,
    true_positive_label: int = 1,
) -> tuple[list[str], list[int]]:

    indices = get_trouble_makers_indices(
        data if channel is None else data[:, channel],
        labels,
        keep_percentile=keep_percentile,
        true_positive_label=true_positive_label,
    )

    return [patient_ids[i] for i in indices.reshape(-1)], indices.reshape(-1).tolist()


def filter_out_trouble_makers(
    data: np.ndarray,
    labels: np.ndarray,
    channel: int | Iterable[int] = 0,
    keep_percentile: int = 95,
    true_positive_label: int = 1,
) -> tuple[np.ndarray, np.ndarray]:

    _data = deepcopy(data)
    _labels = deepcopy(labels)

    trouble_makers = get_trouble_makers_indices(
        _data,
        _labels,
        keep_percentile=keep_percentile,
        true_positive_label=true_positive_label,
    )

    if _data.ndim == 1:
        channel = 0

    indices_to_remove = set(trouble_makers[:, channel].reshape(-1).tolist())
    indices_to_keep = np.array(
        [i for i in range(len(_data)) if i not in indices_to_remove]
    )

    return _data[indices_to_keep, ...], _labels[indices_to_keep, ...]
