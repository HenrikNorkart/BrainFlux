# %%
import imp
import matplotlib.pyplot as plt

from brainflux.aggregators import AggregatedFilterResult
from brainflux.evaluators.evaluator_base import EvaluatorBase
import pandas as pd


class FilterScoreDistributionsEvaluator(EvaluatorBase):
    def _evaluate(self, data: AggregatedFilterResult) -> None:

        fig, axs = self._subplot_init(
            data,
            xlabel="Filter Score",
            ylabel="Count",
        )

        labels = self._label_setup(data)

        for i in range(data.distribution.shape[1]):
            for j in range(data.distribution.shape[2]):
                dist_column = data.distribution[:, i, j]

                # Create histogram data
                bins = 50

                ax: plt.Axes = axs[max(i, j)] if len(axs.shape) == 1 else axs[i, j]

                for ii, (k, v) in enumerate(labels.items()):

                    data_to_plot = dist_column[data.labels == k]
                    count = len(dist_column[data.labels == k])
                    if len(data_to_plot) > 0:
                        ax.hist(
                            data_to_plot,
                            bins=bins,
                            alpha=0.6,
                            color=v["color"],
                            density=False,
                            label=f"{v['name']} (n={count})",
                        )

                        mean_value = data_to_plot.mean()
                        ax.axvline(
                            mean_value,
                            color=v["color"],
                            linestyle="--",
                            linewidth=2,
                            alpha=0.8,
                        )
                        ax.text(
                            0.02,
                            0.98 - (0.1 * ii),
                            f"{v['name']} Mean: {mean_value:.3f}",
                            transform=ax.transAxes,
                            verticalalignment="top",
                            bbox=dict(
                                boxstyle="round", facecolor=v["color"], alpha=0.3
                            ),
                        )

                ax.set_yscale("log")
                ax.grid(True)

        ledgers = []
        for k, v in labels.items():
            ledgers.extend([f"{v['name']}", f"{v['name']} mean"])

        fig.legend(ledgers, loc="upper right")

        plt.tight_layout()


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

    fsde = FilterScoreDistributionsEvaluator(
        show=False,
        save=True,
        block=False,
    )
    fsde.evaluate(res)
