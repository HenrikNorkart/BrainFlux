from __future__ import annotations

from functools import wraps
from pathlib import Path
import pickle
import os

import numpy as np
from tqdm import tqdm

from brainflux.dataloaders.base_loader import BaseLoader
from brainflux.filters.base_filter import BaseFilter
from brainflux.dataclasses import AggregatedFilterResult
from brainflux.utils import load_dotenv

load_dotenv()

BASE_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / os.getenv(
    "CACHE_DIR", ".cache"
)
USE_CACHED_DATA = os.getenv("USE_CACHED_DATA", "True").lower() in ("true", "1", "yes")


class FilterAggregator:
    def __init__(
        self,
        loader: BaseLoader,
        data_filter: BaseFilter,
    ):
        # assert loader.has_labels, "Loader must have labels to use Aggregator."

        self._loader = loader
        self._filter: BaseFilter = data_filter
        self._data_path = data_filter.data_source.data_path

    @property
    def cache_file(self) -> Path:
        return (
            BASE_CACHE_DIR
            / f"{Path(self._data_path).name}_{str(self._filter)}_{self._loader.labels_file_name}"
        ).with_suffix(".pkl")

    def call_post_process(func):
        @wraps(func)
        def wrapper(self: FilterAggregator, *args, **kwds):
            results: AggregatedFilterResult = func(self, *args, **kwds)
            results.distribution = self._filter.post_process_distribution(
                results.distribution,
                results.labels,
            )
            return results

        return wrapper

    @call_post_process
    def aggregate(self, *, use_cache: bool | None = None) -> AggregatedFilterResult:

        if use_cache is None:
            use_cache = USE_CACHED_DATA

        if self._loader.is_in_dev_mode:
            use_cache = False

        # Try to load from cache
        if use_cache:
            if self.cache_file.exists():
                print(f"Using cached data from: {self.cache_file}")
                try:
                    with open(self.cache_file, "rb") as f:
                        results = pickle.load(f)
                    if results is not None:
                        return results
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
