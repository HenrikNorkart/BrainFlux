from abc import ABC, abstractmethod
import random

import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from pathlib import Path

from brainflux.aggregators import AggregatedFilterResult
from brainflux.filters.base_filter import BaseFilter

load_dotenv()


SAVE_DIR = Path(os.getenv("FIGURE_SAVE_DIR", "./figures")).resolve()
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


class EvaluatorBase(ABC):

    def __init__(
        self,
        block: bool = True,
        save: bool = False,
        show: bool = True,
        plot_title: str = f"Results",
    ) -> None:
        super().__init__()
        self._block = block
        self._save = save
        self._show = show
        self._plot_title = plot_title

    @property
    def plot_title(self) -> str:
        return self._plot_title

    def _data_formatting(self, data: AggregatedFilterResult) -> AggregatedFilterResult:

        data_format = data.distribution.shape[1:]
        if len(data_format) == 1:
            data.distribution = data.distribution.reshape(-1, data_format[0], 1)
        elif len(data_format) > 2:
            raise ValueError("Data format not supported.")

    def _subplot_init(
        self, data: AggregatedFilterResult, xlabel: str, ylabel: str
    ) -> tuple[plt.Figure, plt.Axes]:
        h_channels = data.distribution.shape[2]
        v_channels = data.distribution.shape[1]
        fig, axs = plt.subplots(
            v_channels,
            h_channels,
            figsize=(16, 9),
            sharex=True,
            sharey=True,
        )
        fig.suptitle(self._plot_title, fontsize=16)

        for j in range(axs.shape[0]):
            if len(axs.shape) > 1:
                axs[-j - 1, 0].set_ylabel(ylabel)
            else:
                axs[-j - 1].set_ylabel(ylabel)
        if len(axs.shape) > 1:
            for i in range(axs.shape[1]):
                axs[-1, i].set_xlabel(f"{xlabel}")
        else:
            axs[-1].set_xlabel(xlabel)

        return fig, axs

    @staticmethod
    def _add_subplot_labels(axs: plt.Axes, xlabel: str, ylabel: str) -> None:
        print("DEPRECATED WARNING: Use _subplot_init instead.")
        for j in range(axs.shape[0]):
            if len(axs.shape) > 1:
                axs[-j - 1, 0].set_ylabel(ylabel)
            else:
                axs[-j - 1].set_ylabel(ylabel)
        if len(axs.shape) > 1:
            for i in range(axs.shape[1]):
                axs[-1, i].set_xlabel(f"{xlabel}\nT{i}")
        else:
            axs[-1].set_xlabel(xlabel)

    @staticmethod
    def _label_setup(data: AggregatedFilterResult) -> dict[int, dict[str, str]]:
        labels = {
            l: {
                "name": f"Class {l}",
                "color": "#{:06x}".format(random.randint(0, 0xFFFFFF)),
            }
            for l in set(data.labels)
        }

        if 0 in labels and 1 in labels:
            # Ensure consistent colors for binary classification
            labels[0]["name"] = "Non-Survivor"
            labels[0]["color"] = "red"
            labels[1]["name"] = "Survivor"
            labels[1]["color"] = "green"

        return labels

    def _data_validation(self, data: AggregatedFilterResult) -> None:
        pass

    @abstractmethod
    def _evaluate(self, data: AggregatedFilterResult) -> None:
        pass

    def evaluate(self, data: AggregatedFilterResult) -> None:

        self._data_validation(data)
        self._data_formatting(data)

        self._evaluate(data)

        if self._save:
            plt.savefig(SAVE_DIR / f"{self._plot_title}.png", dpi=300)

        if self._show:
            plt.show(block=self._block)
