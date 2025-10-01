from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import pandas as pd

from brainflux.dataclasses import EEGData, DataMeta


class BaseLoader(ABC):

    def __init__(
        self, label_file: str | Path | DataMeta | None = None, *, dev_mode: bool = False
    ):
        self._label_file = label_file
        self._dev_mode = dev_mode

        if self._label_file is None:
            self._has_labels = False
            print("No label file provided. Loader will not load labels.")
            return

        self._has_labels = True

        if isinstance(self._label_file, str):
            self._label_file = Path(self._label_file)

        elif isinstance(self._label_file, DataMeta):
            self._label_file = self._label_file.data_path

        assert (
            self._label_file.exists()
        ), f"Label file {self._label_file} does not exist."
        assert (
            self._label_file.suffix == ".csv"
        ), f"Label file {self._label_file} is not a .csv file."

        self.labels_df = pd.read_csv(self._label_file)

        self._all_labels_to_find = None

    @property
    def is_in_dev_mode(self) -> bool:
        return self._dev_mode

    @property
    def has_labels(self) -> bool:
        return self._has_labels

    @property
    def labels_file_name(self) -> str | None:
        return self._label_file.name or "NoLabels"

    def _path_extract_patient_id(self, file_path: Path) -> str:
        """
        Extract patient ID from the file path.

        Args:
            file_path (Path): The path to the file.

        Returns:
            str: The extracted patient ID.
        """
        return file_path.stem.split("_")[0]

    def _path_extract_session_id(self, file_path: Path) -> int:
        """
        Extract session ID from the file path.

        Args:
            file_path (Path): The path to the file.

        Returns:
            str: The extracted session ID.
        """
        return int(file_path.stem.split("_")[1])

    def _path_extract_channel_name(self, file_path: Path) -> str:
        """
        Extract channel name from the file path.

        Args:
            file_path (Path): The path to the file.

        Returns:
            str: The extracted channel name.
        """
        return file_path.parent.name

    def _populate_metadata(self, file_path: Path, eeg_data: EEGData) -> EEGData:
        """
        Populate metadata in the EEGData object.

        Args:
            file_path (Path): The path to the file.
            eeg_data (EEGData): The EEGData object to populate.

        Returns:
            EEGData: The populated EEGData object.
        """
        eeg_data.subject_id = self._path_extract_patient_id(file_path)
        eeg_data.session_id = set([self._path_extract_session_id(file_path)])
        eeg_data.channel_names = self._path_extract_channel_name(file_path)
        return eeg_data

    def _get_file_label(self, file_path: Path) -> int | None:
        if not self._label_file:
            return -1  # No labels, return dummy label

        patient_id = self._path_extract_patient_id(file_path)

        match = self.labels_df[self.labels_df["patient"] == patient_id]
        if not match.empty:
            label = match["label"].values[0]
            if self._all_labels_to_find and patient_id in self._all_labels_to_find:
                self._all_labels_to_find.remove(patient_id)
            return label

        return None

    def load(self, file_path: str | Path) -> EEGData | None:
        """
        Load data from the given file path.

        Args:
            file_path (str): The path to the file.

        Returns:
            EEGData: The loaded EEGData object.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        assert file_path.exists(), f"File {file_path} does not exist."

        if (label := self._get_file_label(file_path)) is None:
            return None

        data = self._load(file_path)
        self._populate_metadata(file_path, data)
        data.label = label
        return data

    def load_directory(self, dir_path: str | Path) -> Iterator[EEGData]:
        """
        Lazy load all data files from the given directory.

        Args:
            dir_path (str): The path to the directory.

        Returns:
            Iterator[EEGData]: An iterator of loaded EEGData objects.
        """

        if self.has_labels:
            self._all_labels_to_find = set(self.labels_df["patient"].tolist())

        if isinstance(dir_path, str):
            dir_path = Path(dir_path)

        assert (
            dir_path.exists() and dir_path.is_dir()
        ), f"Directory {dir_path} does not exist or is not a directory."

        current_patient_id = None
        aggregated_result = None

        for i, file_path in enumerate(dir_path.glob("**/*")):

            if current_patient_id is None:
                current_patient_id = self._path_extract_patient_id(file_path)

            elif self._path_extract_patient_id(file_path) != current_patient_id:
                if aggregated_result is not None:
                    yield aggregated_result
                aggregated_result = None
                current_patient_id = self._path_extract_patient_id(file_path)

            if file_path.is_file():
                try:
                    if (result := self.load(file_path)) is not None:
                        if aggregated_result is None:
                            aggregated_result = result
                        else:
                            aggregated_result += result
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

            if self._dev_mode and i >= 50:  # Limit to first 10 files for dev mode
                break

        if aggregated_result is not None:
            yield aggregated_result

        if self.has_labels and self._all_labels_to_find is not None:
            if self._all_labels_to_find and len(self._all_labels_to_find) > 0:
                print(
                    f"Warning: {len(self._all_labels_to_find)} labels were not found in the data directory"
                )

    @abstractmethod
    def _load(self, file_path: Path) -> EEGData:
        pass
