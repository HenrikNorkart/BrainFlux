from os import path
from pathlib import Path

import numpy as np

from brainflux.dataloaders.base_loader import BaseLoader
from brainflux.dataclasses import EEGData, DataMeta


class NumpyLoader(BaseLoader):

    def __init__(
        self, label_file: str | Path | DataMeta | None = None, *, dev_mode: bool = False
    ):
        super().__init__(label_file=label_file, dev_mode=dev_mode)

    def _load(self, file_path: Path):

        assert file_path.suffix == ".npy", f"File {file_path} is not a .npy file."

        file_path = Path(file_path)
        data = np.load(file_path)

        eeg_data = EEGData(
            data=data,
        )

        return eeg_data
