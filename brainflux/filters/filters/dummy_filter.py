from dataclasses import dataclass
from brainflux.dataclasses import DataMeta, EEGData
from brainflux.filters.base_filter import BaseFilter, BaseFilterResult

import numpy as np


@dataclass
class DummyFilterResult(BaseFilterResult):
    data_shape: tuple[int, ...]

    def __repr__(self) -> str:
        return f"Dummy Filter Result. Shape: {self.data_shape}"


class DummyFilter(BaseFilter):

    def __init__(self, data_source: DataMeta, data_shape: tuple[int, ...] = (1,)):
        super().__init__(data_source)
        self._data_shape = data_shape

    def _apply(self, eeg_data: EEGData) -> BaseFilterResult:
        random_distribution = np.random.rand(*self._data_shape)
        return DummyFilterResult(
            distribution=random_distribution,
            data_meta=self.data_source,
            data_shape=self._data_shape,
        )
