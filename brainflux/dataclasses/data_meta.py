from pathlib import Path

from dataclasses import dataclass


@dataclass(frozen=True)
class DataMeta:
    name: str
    data_path: Path
    data_max_value: float
    data_min_value: float
