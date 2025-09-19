from functools import reduce

from brainflux.aggregators.filter_aggregator import (
    FilterAggregator,
    AggregatedFilterResult,
)


class EnsembleFilterAggregator:
    def __init__(self, aggregators: list[FilterAggregator]):
        self._aggregators = aggregators

    def aggregate(self, *, load_from_cache: bool = True) -> AggregatedFilterResult:

        # Parallel processing could be added here if needed
        aggregated_results = [
            agg.aggregate(use_cache=load_from_cache) for agg in self._aggregators
        ]
        return reduce(lambda x, y: x | y, aggregated_results)
