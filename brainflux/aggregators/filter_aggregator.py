from pathlib import Path
from dataclasses import dataclass, field
import pickle
import os

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from brainflux.dataloaders.base_loader import BaseLoader
from brainflux.filters.base_filter import BaseFilter


load_dotenv()
BASE_CACHE_DIR = Path(os.getenv("CACHE_DIR", ".")).resolve()
USE_CACHED_DATA = os.getenv("USE_CACHED_DATA", "True").lower() in ("true", "1", "yes")


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


class FilterAggregator:
    def __init__(
        self,
        loader: BaseLoader,
        data_filter: BaseFilter,
    ):
        # assert loader.has_labels, "Loader must have labels to use Aggregator."

        self._loader = loader
        self._filter = data_filter
        self._data_path = data_filter.data_source.data_path

    @property
    def cache_file(self) -> Path:
        return (
            BASE_CACHE_DIR
            / f"{Path(self._data_path).name}_{str(self._filter)}_{self._loader.labels_file_name}"
        ).with_suffix(".pkl")

    def aggregate(self, *, use_cache: bool | None = None) -> AggregatedFilterResult:

        if use_cache is None:
            use_cache = USE_CACHED_DATA

        # Try to load from cache
        if use_cache:
            if self.cache_file.exists():
                print(f"Using cached data from: {self.cache_file}")
                try:
                    with open(self.cache_file, "rb") as f:
                        res = pickle.load(f)
                    if res is not None:
                        return res
                    else:
                        print(f"Cached file is None, re-aggregating.")
                        self.cache_file.unlink()

                except Exception as e:
                    print(f"Failed to load cached aggregation: {e}")
            else:
                print(f"No cached data found at: {self.cache_file}")
                print(f"Aggregating data from scratch.")

        results = self._process()

        # Save to cache if needed
        if use_cache:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file.open("wb") as f:
                f.write(pickle.dumps(results))

        return results

    def _process(self) -> AggregatedFilterResult:

        eeg_data_list = self._loader.load_directory(self._data_path)

        known_patient_ids = []
        aggregated_data = None
        labels = None

        itr = tqdm(
            eeg_data_list,
            desc=f"Aggregating Filter Results ({self._filter.data_source.name})",
        )
        for data in itr:

            if data is None:
                print(f"Skipping data that is None")
                continue

            if data.subject_id in known_patient_ids:
                print(f"Skipping duplicate patient ID: {data.subject_id}")
                continue
            known_patient_ids.append(data.subject_id)

            filtered_data = self._filter.apply(data)

            if aggregated_data is None:
                aggregated_data = filtered_data.distribution.reshape(
                    1, *filtered_data.distribution.shape
                )
            else:
                aggregated_data = np.vstack(
                    (
                        aggregated_data,
                        filtered_data.distribution.reshape(
                            1, *filtered_data.distribution.shape
                        ),
                    )
                )
            if labels is None:
                labels = np.array([data.label])
            else:
                labels = np.append(labels, data.label)

        if aggregated_data.shape[0] != len(labels):
            raise ValueError(
                "Number of aggregated samples does not match number of labels."
            )
        if len(aggregated_data.shape) == 2:
            aggregated_data = aggregated_data.reshape(
                aggregated_data.shape[0], aggregated_data.shape[1], 1
            )

        return AggregatedFilterResult(
            distribution=aggregated_data,
            labels=labels,
            patient_ids=known_patient_ids,
            data_filter=self._filter,
        )

    def __call__(self, *args, **kwds):
        return self.aggregate(*args, **kwds)
