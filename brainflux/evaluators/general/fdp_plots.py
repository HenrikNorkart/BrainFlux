from cProfile import label
import numpy as np
import matplotlib.pyplot as plt


from brainflux.aggregators import AggregatedFilterResult
from brainflux.evaluators.evaluator_base import EvaluatorBase
from brainflux.filters.base_filter import BaseFilter


class FDPPlotsEvaluator(EvaluatorBase):
    def _evaluate(self, data: AggregatedFilterResult) -> None:

        fig, axs = self._subplot_init(
            data,
            xlabel="Filter Score",
            ylabel="Percentile",
        )

        def order_data(data):
            filter_scores = np.linspace(0, 1, 500)
            percentages = np.array(
                [(data >= score).sum() / len(data) for score in filter_scores]
            )
            mask = percentages > 0
            return filter_scores[mask], percentages[mask]

        labels = self._label_setup(data)

        for i in range(data.distribution.shape[1]):
            for j in range(data.distribution.shape[2]):
                dist_column = data.distribution[:, i, j]
                ax = axs[max(i, j)] if len(axs.shape) == 1 else axs[i, j]

                # Should have ledgers
                for l in labels.keys():
                    data_l_ordered, percentages_l = order_data(
                        dist_column[data.labels == l]
                    )

                    ax.plot(data_l_ordered, percentages_l, color=labels[l]["color"])

                    ax.axvline(
                        x=np.max(data_l_ordered),
                        color=labels[l]["color"],
                        linestyle="--",
                        linewidth=3,
                    )

                # Should not have ledgers
                for l in labels.keys():
                    data_l_ordered, percentages_l = order_data(
                        dist_column[data.labels == l]
                    )
                    ax.fill_between(
                        data_l_ordered,
                        percentages_l,
                        color=labels[l]["color"],
                        alpha=0.2,
                    )
                    ax.scatter(
                        data_l_ordered, percentages_l, color=labels[l]["color"], s=10
                    )

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.grid(True)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.15)

        ledgers = []
        for l in labels.keys():
            ledgers += [labels[l]["name"], f"{labels[l]['name']} Zero Line"]

        fig.legend(ledgers, loc="upper right")

    def _data_validation_wrapper(func):
        def inner(self, data: AggregatedFilterResult) -> None:
            if not isinstance(data.data_filter, (BaseFilter, list)):
                raise ValueError(
                    "data_filter must be a BaseFilter or a list of BaseFilters."
                )
            if isinstance(data.data_filter, list):
                for df in data.data_filter:
                    if not isinstance(df, BaseFilter):
                        raise ValueError(
                            "All elements in data_filter list must be BaseFilter instances."
                        )
            # if len(set(data.labels)) != 2:
            #     raise ValueError("FDP plots only support binary classification.")

            func(self, data)

        return inner
