# %%

import numpy as np

from brainflux.dataclasses import DataMeta
from brainflux.filters.filters.range_filter import RangeFilter


class RangeFilterTFIDF(RangeFilter):
    def __init__(
        self,
        data_source: DataMeta,
        num_ranges: int = 8,
        num_time_divisions: int = 1,
        min_filter_score_cutoff: float = 0.1,
        true_label: int | None = 1,
    ):
        super().__init__(
            data_source, num_ranges=num_ranges, num_time_divisions=num_time_divisions
        )
        self._name = "Range Filter with TF-IDF"
        self._min_filter_score_cutoff = min_filter_score_cutoff
        self._true_label = true_label

    def __repr__(self):
        return f"RangeFilterTFIDF_{self._num_ranges}x{self._num_time_divisions}@{self._min_filter_score_cutoff}"

    def post_process_distribution(
        self, aggregated_distribution: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:

        if self._true_label is None:
            idx = np.arange(labels.shape[0])

        else:
            idx = labels == self._true_label

        count = np.sum(
            aggregated_distribution[idx] > self._min_filter_score_cutoff,
            axis=0,
        )

        idf = np.log((1 + aggregated_distribution[idx].shape[0]) / (1 + count))

        aggregated_distribution *= idf

        aggregated_distribution /= np.sum(
            aggregated_distribution, axis=1, keepdims=True
        )

        return aggregated_distribution


if __name__ == "__main__":
    from brainflux.dataloaders import NumpyLoader
    from brainflux.aggregators import FilterAggregator
    from brainflux.dataclasses import suppression_ratio

    data_loader = NumpyLoader(
        label_file="/workspaces/BrainFlux/test_data/train.csv", dev_mode=True
    )

    data_filter = RangeFilterTFIDF(
        data_source=suppression_ratio,
        min_filter_score_cutoff=0.05,
        num_ranges=4,
        num_time_divisions=4,
    )
    res = FilterAggregator(loader=data_loader, data_filter=data_filter).aggregate(
        use_cache=False
    )
