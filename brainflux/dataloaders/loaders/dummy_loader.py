from pathlib import Path

import numpy as np
from pyparsing import Iterator

from brainflux.dataloaders.base_loader import BaseLoader
from brainflux.dataclasses.eeg import EEGData

from uuid import uuid4


class DummyLoader(BaseLoader):

    def __init__(
        self,
        length_of_data: int = 5120,
        min_data_value: float = 0.0,
        max_data_value: float = 1.0,
        num_patients: int = 500,
        num_classes: int = 2,
        label_file: str | Path | None = None,
        dev_mode: bool = False,
    ):
        super().__init__(label_file=label_file, dev_mode=dev_mode)
        self._length_of_data = length_of_data
        self._min_data_value = min_data_value
        self._max_data_value = max_data_value
        self._num_patients = num_patients
        self._num_classes = num_classes

        self._counter = 0

    def _load(self, *args, **kwargs):

        return EEGData(
            data=np.random.rand(self._length_of_data)
            * (self._max_data_value - self._min_data_value)
            + self._min_data_value,
            sampling_rate=256,
            subject_id=f"TEST-{str(int(uuid4()))[:4]}-{str(int(uuid4()))[:3]}",
            label=np.random.randint(0, self._num_classes),
        )

    def load_directory(self, *args, **kwargs) -> Iterator[EEGData]:
        num_files = 20 if self._dev_mode else self._num_patients
        for _ in range(num_files):
            yield self._load()
