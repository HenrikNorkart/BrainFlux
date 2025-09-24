# %%
from dataclasses import dataclass, field
from typing import Literal, Tuple


from brainflux.evaluators.evaluator_base import EvaluatorBase
from brainflux.aggregators.filter_aggregator import AggregatedFilterResult
from brainflux.utils.troublemaker import (
    get_trouble_makers_patient_ids,
    unlock_power,
    intruder_power,
)
from brainflux.utils.filter_discriminative_power import filter_discriminative_power


from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
import numpy as np
import pandas as pd

OverUnder = Literal["over", "under"]


@dataclass
class TroubleMaker:
    patient_id: str
    index: int
    z_scores: list[float] = field(default_factory=list)
    fdp_scores: list[float] = field(default_factory=list)
    unlocked_power: list[float] = field(default_factory=list)
    intruder_score: list[float] = field(default_factory=list)
    lof_score: float | None = None
    mahalanobis_score: float | None = None
    channels: list[Tuple[int, ...]] = field(default_factory=list)

    @property
    def appearances(self) -> int:
        return len(self.channels)

    @property
    def avg_z_score(self) -> float | None:
        if self.z_scores and self.appearances > 0:
            return sum(self.z_scores) / self.appearances
        return None

    @property
    def max_z_score(self) -> float | None:
        if self.z_scores:
            return max(self.z_scores)
        return None

    @property
    def total_z_score(self) -> float:
        return sum(self.z_scores)

    @property
    def avg_fdp_score(self) -> float | None:
        if self.fdp_scores and self.appearances > 0:
            return sum(self.fdp_scores) / self.appearances
        return None

    @property
    def max_fdp_score(self) -> float | None:
        if self.fdp_scores:
            return max(self.fdp_scores)
        return None

    @property
    def total_fdp_score(self) -> float:
        return sum(self.fdp_scores)

    @property
    def avg_unlocked_power(self) -> float | None:
        if self.unlocked_power and self.appearances > 0:
            return sum(self.unlocked_power) / self.appearances
        return None

    @property
    def max_unlocked_power(self) -> float | None:
        if self.unlocked_power:
            return max(self.unlocked_power)
        return None

    @property
    def total_unlocked_power(self) -> float:
        return sum(self.unlocked_power)

    @property
    def avg_intruder_score(self) -> float | None:
        if self.intruder_score and self.appearances > 0:
            return sum(self.intruder_score) / self.appearances
        return None

    @property
    def max_intruder_score(self) -> float | None:
        if self.intruder_score:
            return max(self.intruder_score)
        return None

    @property
    def total_intruder_score(self) -> float:
        return sum(self.intruder_score)

    def __dict__(self):
        return {
            "patient_id": self.patient_id,
            "index": self.index,
            "avg_z_score": self.avg_z_score,
            "max_z_score": self.max_z_score,
            "total_z_score": self.total_z_score,
            "avg_fdp_score": self.avg_fdp_score,
            "max_fdp_score": self.max_fdp_score,
            "total_fdp_score": self.total_fdp_score,
            "avg_unlocked_power": self.avg_unlocked_power,
            "max_unlocked_power": self.max_unlocked_power,
            "total_unlocked_power": self.total_unlocked_power,
            "avg_intruder_score": self.avg_intruder_score,
            "max_intruder_score": self.max_intruder_score,
            "total_intruder_score": self.total_intruder_score,
            "lof_score": self.lof_score,
            "mahalanobis_score": self.mahalanobis_score,
            "appearances": self.appearances,
        }

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__()])


class TroubleMakerTracker(EvaluatorBase):

    def __init__(
        self,
        block=True,
        plot_title: str | None = None,
        true_labels: int = 1,
        over_or_under: OverUnder = "over",
        keep_percentile: int = 95,
        save: bool = False,
        show: bool = True,
    ):
        super().__init__(
            block,
            save,
            show,
            plot_title or f"Trouble maker impact for label {true_labels}",
        )
        self._true_label = true_labels
        self._over_or_under = over_or_under
        self._keep_percentile = keep_percentile

    def _evaluate(self, data: AggregatedFilterResult) -> list[TroubleMaker]:

        troublemakers_set: list[TroubleMaker] = []

        def z_score(x: np.ndarray, data_point: float) -> float:
            return abs(data_point - x.mean()) / x.std()

        for i in range(data.distribution.shape[1]):
            for j in range(data.distribution.shape[2]):

                dist_column = data.distribution[:, i, j]
                labels = data.labels

                troublemakers, tm_indices = get_trouble_makers_patient_ids(
                    dist_column,
                    labels,
                    patient_ids=data.patient_ids,
                    true_positive_label=self._true_label,
                    keep_percentile=self._keep_percentile,
                )

                fdp = filter_discriminative_power(
                    dist_column,
                    labels,
                    non_zero=True,
                )

                unlock_powers = unlock_power(
                    dist_column,
                    labels,
                    patient_ids=data.patient_ids,
                    true_positive_label=self._true_label,
                    keep_percentile=self._keep_percentile,
                )

                intruder_powers = intruder_power(
                    dist_column,
                    labels,
                    patient_ids=data.patient_ids,
                    true_positive_label=self._true_label,
                    keep_percentile=self._keep_percentile,
                )

                dist_column_true_label = dist_column[labels == self._true_label]

                for tm_patient_id, tm_index in zip(troublemakers, tm_indices):

                    for tms in troublemakers_set:
                        if tm_patient_id == tms.patient_id:
                            tms.channels.append((i, j))
                            tms.z_scores.append(
                                z_score(dist_column_true_label, dist_column[tm_index])
                            )
                            tms.fdp_scores.append(fdp)
                            tms.unlocked_power.append(unlock_powers[tm_patient_id])
                            tms.intruder_score.append(intruder_powers[tm_patient_id])
                            break

                    else:
                        troublemakers_set.append(
                            TroubleMaker(
                                patient_id=tm_patient_id,
                                z_scores=[
                                    z_score(
                                        dist_column_true_label, dist_column[tm_index]
                                    )
                                ],
                                fdp_scores=[fdp],
                                unlocked_power=[unlock_powers[tm_patient_id]],
                                intruder_score=[intruder_powers[tm_patient_id]],
                                channels=[(i, j)],
                                index=tm_index,
                            )
                        )

        detector_lof = LocalOutlierFactor()
        detector_mahalanobis = MinCovDet()

        for tms in troublemakers_set:

            data_local = data.distribution[labels == self._true_label]

            label_mask = np.where(data.labels == self._true_label, 1, 0)

            new_index = sum(label_mask[: tms.index + 1]) - 1

            selected_channels = np.stack(
                [data_local[:, i, j].reshape(-1, 1) for (i, j) in tms.channels], axis=-1
            ).reshape(-1, tms.appearances)

            detector_lof.fit(selected_channels)

            detector_mahalanobis.fit(selected_channels)
            tms.lof_score = detector_lof.negative_outlier_factor_[new_index] * -1
            tms.mahalanobis_score = detector_mahalanobis.mahalanobis(
                selected_channels.reshape(-1, tms.appearances)
            )[new_index]

        return troublemakers_set


if __name__ == "__main__":
    from brainflux.dataloaders import NumpyLoader
    from brainflux.filters import RangeFilter
    from brainflux.aggregators import FilterAggregator
    from brainflux.dataclasses import suppression_ratio

    data_loader = NumpyLoader(label_file="/workspaces/BrainFlux/test_data/train.csv")

    data_filter = RangeFilter(
        data_source=suppression_ratio, num_ranges=4, num_time_divisions=4
    )

    res = FilterAggregator(loader=data_loader, data_filter=data_filter).aggregate()

    tmt = TroubleMakerTracker(
        block=False,
        true_labels=0,
        over_or_under="over",
        keep_percentile=95,
        save=True,
        show=True,
    )
    tms = tmt.evaluate(res)
    df = pd.DataFrame([tm.__dict__() for tm in tms]).sort_values(
        by=["avg_fdp_score", "avg_z_score"], ascending=False
    )
    print(df)
