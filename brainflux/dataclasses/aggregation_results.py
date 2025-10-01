from brainflux.filters.base_filter import BaseFilter
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AggregatedFilterResult:
    distribution: np.ndarray
    labels: np.ndarray
    patient_ids: list[str] = field(default_factory=list)
    data_filter: BaseFilter | list[BaseFilter] | None = None

    def __or__(self, value):
        if not isinstance(value, AggregatedFilterResult):
            return NotImplemented

        # Find common patient IDs between self and value
        common_patient_ids = list(set(self.patient_ids) & set(value.patient_ids))
        if not common_patient_ids:
            raise ValueError("No common patient IDs found between the two results")

        # Get indices for common patient IDs in both results
        self_indices = [
            i for i, pid in enumerate(self.patient_ids) if pid in common_patient_ids
        ]
        value_indices = [
            i for i, pid in enumerate(value.patient_ids) if pid in common_patient_ids
        ]

        num_indices = len(self_indices)

        # Filter distributions and labels to only include common patients
        self_filtered_distribution = self.distribution[self_indices].reshape(
            num_indices, -1
        )
        self_filtered_labels = self.labels[self_indices]
        value_filtered_distribution = value.distribution[value_indices].reshape(
            num_indices, -1
        )
        value_filtered_labels = value.labels[value_indices]

        for l1, l2 in zip(self_filtered_labels, value_filtered_labels):
            assert l1 != l2, "Labels do not match for common patient IDs"

        # Ensure both arrays have the same number of samples (rows)
        assert (
            self_filtered_distribution.shape[0] == value_filtered_distribution.shape[0]
        ), "Both distributions must have the same number of samples"

        combined_distribution = np.concatenate(
            (self_filtered_distribution, value_filtered_distribution), axis=1
        )

        assert combined_distribution.shape[0] == len(
            self.labels
        ), "Number of samples and labels must match"
        assert combined_distribution.ndim == 2, "Combined distribution must be 2D"
        assert len(self.patient_ids) == len(
            self.labels
        ), "Number of patient IDs and labels must match"

        if isinstance(self.data_filter, BaseFilter) and isinstance(
            value.data_filter, BaseFilter
        ):
            data_filter = [self.data_filter, value.data_filter]
        elif isinstance(self.data_filter, list) and isinstance(value.data_filter, list):
            data_filter = self.data_filter + value.data_filter
        elif isinstance(self.data_filter, BaseFilter) and isinstance(
            value.data_filter, list
        ):
            data_filter = [self.data_filter] + value.data_filter
        elif isinstance(self.data_filter, list) and isinstance(
            value.data_filter, BaseFilter
        ):
            data_filter = self.data_filter + [value.data_filter]
        else:
            data_filter = None

        return AggregatedFilterResult(
            distribution=combined_distribution,
            labels=self.labels,
            patient_ids=self.patient_ids,
            data_filter=data_filter,
        )
