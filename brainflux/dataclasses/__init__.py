from pathlib import Path
import os

from brainflux.dataclasses.data_meta import DataMeta
from brainflux.dataclasses.eeg import EEGData
from brainflux.dataclasses.aggregation_results import AggregatedFilterResult
from brainflux.dataclasses.attributes import AttributeExplanation


BASE_DATA_PATH = Path(os.getenv("BASE_DATA_PATH", ".")).resolve()


aEEG = DataMeta(
    name="aEEG",
    data_path=BASE_DATA_PATH / "aEEG",
    data_max_value=80.0,
    data_min_value=0.0,
)

suppression_ratio = DataMeta(
    name="suppression_ratio",
    data_path=BASE_DATA_PATH / "suppression_ratio",
    data_max_value=100.0,
    data_min_value=0.0,
)

spike_detection_10_sec = DataMeta(
    name="spike_detection (10 sec)",
    data_path=BASE_DATA_PATH / "spike_detection (10 sec)",
    data_max_value=10.0,
    data_min_value=0.0,
)


data_catalog = {
    "aEEG": aEEG,
    "suppression_ratio": suppression_ratio,
    "spike_detection (10 sec)": spike_detection_10_sec,
}
__all__ = [
    "EEGData",
    "DataMeta",
    "AggregatedFilterResult",
    "data_catalog",
    "aEEG",
    "suppression_ratio",
    "spike_detection_10_sec",
    "AttributeExplanation",
]
