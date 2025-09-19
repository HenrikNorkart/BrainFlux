from brainflux.filters.base_filter import BaseFilter, BaseFilterResult
from brainflux.dataclasses import DataMeta

import numpy as np
from prettytable import PrettyTable

from dataclasses import dataclass


@dataclass
class RangeFilterResult(BaseFilterResult):
    num_ranges: int | None = None
    num_time_divisions: int | None = None

    def __post_init__(self):

        if self.num_ranges is None:
            self.num_ranges = self.distribution.shape[0]
        if self.num_time_divisions is None:
            if self.distribution.ndim == 1:
                self.num_time_divisions = 1
            elif self.distribution.ndim == 2:
                self.num_time_divisions = self.distribution.shape[1]
            else:
                raise ValueError("Invalid distribution shape")

    def __repr__(self) -> str:
        field_names = ["Range"]
        for i in range(self.num_time_divisions):
            field_names.append(f"T{i}")

        table = PrettyTable(field_names)
        table.title = "Filter Scores"

        table.align = "l"

        range_edges = np.linspace(self.min_value, self.max_value, self.num_ranges + 1)
        for i in range(self.num_ranges):
            range_str = f"[{range_edges[-i-2]:.2f}, {range_edges[-i-1]:.2f})"

            if self.num_time_divisions == 1:
                fractions = [f"{self.distribution[-i-1]:.4f}"]
            else:
                fractions = [
                    f"{self.distribution[i][j]:.4f}"
                    for j in range(self.num_time_divisions)
                ]
            table.add_row([range_str, *fractions])
        return str(table)


class RangeFilter(BaseFilter):

    def __init__(
        self,
        data_source: DataMeta,
        num_ranges: int = 8,
        num_time_divisions: int = 1,
    ):
        super().__init__(data_source)
        self._num_ranges = num_ranges
        self._num_time_divisions = num_time_divisions

    @property
    def num_ranges(self) -> int:
        return self._num_ranges

    @property
    def num_time_divisions(self) -> int:
        return self._num_time_divisions

    def __repr__(self):
        return f"RangeFilter_{self._num_ranges}x{self._num_time_divisions}"

    def apply(self, eeg_data) -> RangeFilterResult:
        return super().apply(eeg_data)

    def _apply(self, eeg_data):
        min_v = self._min_value
        max_v = self._max_value
        num_ranges = self._num_ranges
        num_time_divisions = self._num_time_divisions

        # Flatten eeg_data to 1D array if needed
        data = np.asarray(eeg_data.data).flatten()

        # Split data into num_time_divisions
        if num_time_divisions > 1:
            split_data = np.array_split(data, num_time_divisions)
            distributions = []
            for segment in split_data:
                edges = np.linspace(min_v, max_v, num_ranges + 1)
                counts, _ = np.histogram(segment, bins=edges)
                dist = (
                    counts / segment.size if segment.size > 0 else np.zeros(num_ranges)
                )
                distributions.append(dist)
            distribution = np.array(distributions)
        else:
            # Create range edges
            edges = np.linspace(min_v, max_v, num_ranges + 1)

            # Count datapoints in each range
            counts, _ = np.histogram(data, bins=edges)

            # Calculate distribution (fraction in each range)
            distribution = counts / data.size if data.size > 0 else np.zeros(num_ranges)

        distribution = np.flip(
            distribution, axis=-1
        )  # Flip to have highest range first
        distribution = distribution.transpose()

        return RangeFilterResult(
            distribution=distribution,
            data_meta=self.data_source,
            num_ranges=num_ranges,
            num_time_divisions=num_time_divisions,
        )
