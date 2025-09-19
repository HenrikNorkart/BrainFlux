import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score

from brainflux.aggregators import AggregatedFilterResult
from brainflux.evaluators.evaluator_base import EvaluatorBase

from typing import Literal

OverUnder = Literal["over", "under"]


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

        for i in range(data.distribution.shape[1]):
            for j in range(data.distribution.shape[2]):
                dist_column = data.distribution[:, i, j]
                labels = data.labels

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
        plt.tight_layout()

    def _data_validation(self, data: AggregatedFilterResult) -> None:
        if len(set(data.labels)) < 2:
            raise ValueError("Data must contain at least two classes for evaluation.")
