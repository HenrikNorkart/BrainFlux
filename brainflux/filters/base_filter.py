from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from brainflux.dataclasses import DataMeta, EEGData


@dataclass
class BaseFilterResult(ABC):
    distribution: np.ndarray
    data_meta: DataMeta

    @abstractmethod
    def __repr__(self) -> str:
        pass


class BaseFilter(ABC):

    def __init__(self, data_source: DataMeta):
        self._min_value = data_source.data_min_value
        self._max_value = data_source.data_max_value
        self._data_source = data_source

    @property
    def data_source(self) -> DataMeta:
        return self._data_source

    @property
    def min_value(self) -> float:
        return self._min_value

    @property
    def max_value(self) -> float:
        return self._max_value

    def apply(self, eeg_data: EEGData) -> BaseFilterResult:
        return self._apply(eeg_data)

    @abstractmethod
    def _apply(self, eeg_data: EEGData) -> BaseFilterResult:
        pass
