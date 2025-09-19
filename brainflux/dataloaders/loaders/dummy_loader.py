from pathlib import Path

import numpy as np

from brainflux.dataloaders.base_loader import BaseLoader
from brainflux.dataclasses.eeg import EEGData


class DummyLoader(BaseLoader):

    def __init__(
        self,
        length_of_data: int = 5120,
        min_data_value: float = 0.0,
        max_data_value: float = 1.0,
        label_file: str | Path | None = None,
        dev_mode: bool = False,
    ):
        super().__init__(label_file=label_file, dev_mode=dev_mode)
        self._length_of_data = length_of_data
        self._min_data_value = min_data_value
        self._max_data_value = max_data_value

    def _load(self, *args, **kwargs):

        return EEGData(
            data=np.random.rand(self._length_of_data)
            * (self._max_data_value - self._min_data_value)
            + self._min_data_value
        )
