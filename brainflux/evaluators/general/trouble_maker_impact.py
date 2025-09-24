from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score

from brainflux.aggregators import AggregatedFilterResult
from brainflux.evaluators.evaluator_base import EvaluatorBase

from typing import Literal

OverUnder = Literal["over", "under"]


class TroubleMakerImpactEvaluator(EvaluatorBase):

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
            plot_title or f"Trouble maker impact for label {true_labels}",
        )
        self._true_label = true_labels
        self._over_or_under = over_or_under

    def _calculate_scores(self, dist_column, labels):

        # Calculate F1 score for different thresholds
        thresholds = np.linspace(0.01, 0.99, 100)

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
                labels=[self._true_label],
                average="micro",
                zero_division=0.0,
            )
            recall = recall_score(
                labels,
                predictions,
                labels=[self._true_label],
                average="micro",
                zero_division=0.0,
            )
            precision = precision_score(
                labels,
                predictions,
                labels=[self._true_label],
                average="micro",
                zero_division=0.0,
            )

            if precision == 1.0:
                return threshold, f1, recall, precision

        return 0.0, 0.0, 0.0, 0.0

    def _filter_out_trouble_makers(
        self, dist_column, labels, true_positive_label=1, keep_percentile: int = 95
    ):

        # Find indices where labels != true_positive_label
        non_target_indices = np.where(labels != true_positive_label)[0]
        if len(non_target_indices) > 0:
            # Get scores for non-target labels
            non_target_scores = dist_column[non_target_indices]

            non_target_scores_and_indices = list(
                zip(non_target_scores, non_target_indices)
            )
            non_target_scores_and_indices.sort(reverse=True, key=lambda x: x[0])
            sorted_non_target_scores, sorted_non_target_indices = zip(
                *non_target_scores_and_indices
            )

            cut_off = int(
                len(sorted_non_target_scores) * ((100 - keep_percentile) / 100)
            )

            trouble_maker_indices = sorted_non_target_indices[:cut_off]

            indices_to_keep = np.setdiff1d(
                np.arange(len(labels)), trouble_maker_indices
            )
            filtered_dist_column = dist_column[indices_to_keep]
            filtered_labels = labels[indices_to_keep]

            trouble_maker_indices = sorted_non_target_indices[:cut_off]

            return len(trouble_maker_indices), filtered_dist_column, filtered_labels
        else:
            return 0, dist_column, labels

    def _evaluate(self, data: AggregatedFilterResult) -> None:

        fig, axs = self._subplot_init(
            data,
            xlabel="Filter Score",
            ylabel="Metric Score",
        )

        for i in range(data.distribution.shape[1]):
            for j in range(data.distribution.shape[2]):
                dist_column = data.distribution[:, i, j]
                labels = data.labels

                percentiles = []
                f1_scores = []
                recall_scores = []
                precision_scores = []

                start = 100
                stop = 50
                step_size = 1

                for percentile_keep in list(range(start, stop - 1, -step_size)):

                    _, dist_column_filtered, labels_filtered = (
                        self._filter_out_trouble_makers(
                            dist_column,
                            labels,
                            true_positive_label=self._true_label,
                            keep_percentile=percentile_keep,
                        )
                    )

                    __, best_f1, best_recall, best_precision = self._calculate_scores(
                        dist_column_filtered, labels_filtered
                    )

                    percentiles.append(percentile_keep)
                    precision_scores.append(best_precision)

                    f1_scores.append(best_f1)
                    recall_scores.append(best_recall)

                ax = axs[max(i, j)] if len(axs.shape) == 1 else axs[i, j]

                ax.plot(percentiles, recall_scores, label="Recall", color="orange")

                ax.fill_between(
                    percentiles,
                    recall_scores,
                    recall_scores[0],
                    where=np.array(recall_scores) >= recall_scores[0],
                    alpha=0.2,
                    color="orange",
                    interpolate=True,
                )

                first_precision_one_idx = None
                for idx, precision in enumerate(precision_scores):
                    if precision == 1.0:
                        first_precision_one_idx = idx
                        break

                if first_precision_one_idx is not None:
                    ax.axvline(
                        x=percentiles[first_precision_one_idx],
                        color="green",
                        linestyle="--",
                        alpha=0.7,
                        label="First Perfect Precision",
                    )

                # ax.plot(percentiles, precision_scores, label="Precision", color="green")
                ax.set_xlim(start, stop)
                ax.grid(True)

        fig.legend(
            ["Recall", "Recall gain", "Strong filter threshold"],
            loc="upper right",
        )
        plt.ylim(0, 1)

        plt.tight_layout()
