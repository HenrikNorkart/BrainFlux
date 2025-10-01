# %%
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score

from brainflux.aggregators import AggregatedFilterResult
from brainflux.evaluators.evaluator_base import EvaluatorBase

from typing import Literal

OverUnder = Literal["over", "under"]


@dataclass
class ClassifierMetrics:
    f1_score: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    threshold: np.ndarray

    num_total_entities: int
    found_entities: set = field(default_factory=set, init=False)

    @classmethod
    def init_from_data(
        cls, data: AggregatedFilterResult, true_label: int
    ) -> "ClassifierMetrics":
        if data.distribution.ndim == 3:
            shape = data.distribution.shape[1:]
        else:
            shape = data.distribution.shape

        return cls(
            f1_score=np.zeros(shape),
            precision=np.zeros(shape),
            recall=np.zeros(shape),
            threshold=np.zeros(shape),
            num_total_entities=len(data.labels[data.labels == true_label]),
        )

    @property
    def global_recall(self) -> float:
        if self.num_total_entities == 0:
            return 0.0
        return len(self.found_entities) / self.num_total_entities

    def __repr__(self):
        return f"ClassifierMetrics(f1_score=\n{self.f1_score}\n\nprecision=\n{self.precision}\n\nrecall=\n{self.recall}\n\nthreshold=\n{self.threshold})"


class ClassifierStatsEvaluator(EvaluatorBase):

    def __init__(
        self,
        block=True,
        plot_title: str | None = None,
        true_labels: int = 1,
        over_or_under: OverUnder = "over",
        save: bool = False,
        show: bool = True,
    ):
        super().__init__(
            block,
            save,
            show,
            plot_title or f"Classification results for label {true_labels}",
        )
        self._true_labels = true_labels
        self._over_or_under = over_or_under

    def _calculate_scores(
        self,
        dist_column,
        labels,
    ):

        # Calculate F1 score for different thresholds
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        recall_scores = []
        precision_scores = []

        for threshold in thresholds:

            if self._over_or_under == "under":
                predictions = (dist_column >= threshold).astype(int)
            elif self._over_or_under == "over":
                predictions = (dist_column <= threshold).astype(int)
            else:
                raise ValueError("over_or_under must be 'over' or 'under'")

            f1 = f1_score(
                labels,
                predictions,
                labels=[self._true_labels],
                average="micro",
                zero_division=0.0,
            )
            recall = recall_score(
                labels,
                predictions,
                labels=[self._true_labels],
                average="micro",
                zero_division=0.0,
            )
            precision = precision_score(
                labels,
                predictions,
                labels=[self._true_labels],
                average="micro",
                zero_division=0.0,
            )

            f1_scores.append(f1)
            recall_scores.append(recall)
            precision_scores.append(precision)

        return (
            thresholds,
            np.asarray(f1_scores),
            np.asarray(recall_scores),
            np.asarray(precision_scores),
        )

    def _evaluate(self, data: AggregatedFilterResult) -> None:

        fig, axs = self._subplot_init(
            data,
            xlabel="Percentile of non-target labels kept",
            ylabel="Score",
        )

        res = ClassifierMetrics.init_from_data(data, self._true_labels)

        for i in range(data.distribution.shape[1]):
            for j in range(data.distribution.shape[2]):
                dist_column = data.distribution[:, i, j]
                labels = data.labels
                patient_ids = data.patient_ids

                thresholds, f1_scores, recall_scores, precision_scores = (
                    self._calculate_scores(dist_column, labels)
                )

                ax = axs[max(i, j)] if len(axs.shape) == 1 else axs[i, j]

                ax.plot(thresholds, f1_scores, label="F1 Score", color="blue")
                ax.plot(thresholds, recall_scores, label="Recall", color="orange")
                ax.plot(thresholds, precision_scores, label="Precision", color="green")

                # Fill area under precision curve where precision == 1
                precision_mask = precision_scores == 1.0
                if np.any(precision_mask):
                    ax.fill_between(
                        thresholds,
                        0,
                        precision_scores,
                        where=precision_mask,
                        color="green",
                        alpha=0.2,
                        interpolate=True,
                    )

                # Add text annotation for recall values where precision == 1
                if np.any(precision_mask):
                    precision_1_indices = np.where(precision_mask)[0]
                    for idx in precision_1_indices[
                        ::5
                    ]:  # Show every 5th point to avoid clutter
                        recall_val = recall_scores[idx]
                        threshold_val = thresholds[idx]

                        res.f1_score[i, j] = f1_scores[idx]
                        res.precision[i, j] = precision_scores[idx]
                        res.recall[i, j] = recall_scores[idx]
                        res.threshold[i, j] = threshold_val

                        if self._over_or_under == "under":
                            mask = (dist_column <= threshold_val) & (
                                labels == self._true_labels
                            )
                        else:
                            mask = (dist_column >= threshold_val) & (
                                labels == self._true_labels
                            )

                        print(mask.nonzero()[0])

                        for dx in mask.nonzero()[0]:
                            res.found_entities.add(patient_ids[dx])

                        ax.text(
                            0.02,
                            0.98,
                            f"R: {recall_val:.2f}",
                            transform=ax.transAxes,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="orange", alpha=0.3),
                        )

                        ax.axvline(
                            threshold_val,
                            color="orange",
                            linestyle="--",
                            label=f"Optimal recall",
                        )

                        break

                ax.grid(True)

        fig.legend(
            ["F1-score", "Recall", "Precision", "Strong filter region", "Max recall"],
            loc="upper right",
        )

        fig.suptitle(
            self._plot_title
            + f"\nFound {len(res.found_entities)}/{res.num_total_entities} total entities, Global Recall: {res.global_recall:.3f}",
            fontsize=24,
        )
        fig.tight_layout()
        return res

    def _data_validation(self, data: AggregatedFilterResult) -> None:
        if len(set(data.labels)) < 2:
            raise ValueError("Data must contain at least two classes for evaluation.")


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

    fsde = ClassifierStatsEvaluator(
        show=False,
        save=True,
        block=False,
        true_labels=1,
        over_or_under="under",
    )
    res: ClassifierMetrics = fsde.evaluate(res)

    print(res.found_entities)
    print(f"Found {len(res.found_entities)} entities")
    print(f"Total entities: {res.num_total_entities}")
    print(f"Global Recall: {res.global_recall:.3f}")
