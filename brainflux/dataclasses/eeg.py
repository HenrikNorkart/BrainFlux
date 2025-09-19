from cProfile import label
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np


@dataclass
class EEGData:
    data: np.ndarray
    channel_names: Optional[str] = None
    subject_id: Optional[str] = None
    session_id: Optional[Set[int]] = None
    sampling_rate: Optional[int] = None  # Hz
    duration: Optional[float] = None  # seconds
    label: Optional[int] = None
    data_min: Optional[float] = None
    data_max: Optional[float] = None

    def __post_init__(self):
        if self.data_min is None:
            self.data_min = float(np.min(self.data))
        if self.data_max is None:
            self.data_max = float(np.max(self.data))
        if self.sampling_rate and self.data.shape[0]:
            self.duration = self.data.shape[0] / self.sampling_rate

        if self.session_id is None:
            self.session_id = set([])
        if isinstance(self.session_id, int):
            self.session_id = set([self.session_id])

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        return (
            f"EEGData(subject_id={self.subject_id}, session_id={self.session_id}, "
            f"channels={self.channel_names}, samples={len(self)}, "
            f"label={self.label}, "
            f"sampling_rate={self.sampling_rate}, duration={self.duration} "
            f"seconds, data_min={self.data_min:.2f}, data_max={self.data_max:.2f})"
        )

    def __add__(self, other: "EEGData") -> "EEGData":
        if self.channel_names != other.channel_names:
            raise ValueError("Channel names must match to concatenate EEGData.")
        if self.sampling_rate != other.sampling_rate:
            raise ValueError("Sampling rates must match to concatenate EEGData.")
        if self.subject_id != other.subject_id:
            raise ValueError("Subject IDs must match to concatenate EEGData.")

        if len(self.session_id.intersection(other.session_id)) > 0:
            raise ValueError("Session IDs must be disjoint to concatenate EEGData.")

        new_data = np.concatenate((self.data, other.data), axis=0)
        return EEGData(
            data=new_data,
            channel_names=self.channel_names,
            subject_id=self.subject_id,
            session_id=self.session_id.union(other.session_id),
            sampling_rate=self.sampling_rate,
            label=self.label,  # Keep the label of the first segment
        )
