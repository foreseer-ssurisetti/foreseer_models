"""
Factory pattern implementation for algorithm creation.
"""

from typing import Type, Dict
from .base import BaseAlgorithm


class AlgorithmFactory:
    """Factory class for creating algorithm instances."""

    _algorithms: Dict[str, Type[BaseAlgorithm]] = {}

    @classmethod
    def create(cls, algorithm_name: str, **kwargs) -> BaseAlgorithm:
        """Create an algorithm instance by name."""
        # Lazy import to avoid circular dependencies
        cls._load_algorithms()

        if algorithm_name not in cls._algorithms:
            available = list(cls._algorithms.keys())
            raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {available}")

        algorithm_class = cls._algorithms[algorithm_name]
        return algorithm_class(**kwargs)

    @classmethod
    def register(cls, name: str, algorithm_class: Type[BaseAlgorithm]) -> None:
        """Register a new algorithm class."""
        cls._algorithms[name] = algorithm_class

    @classmethod
    def list_algorithms(cls) -> list:
        """List all available algorithms."""
        cls._load_algorithms()
        return list(cls._algorithms.keys())

    @classmethod
    def _load_algorithms(cls):
        """Load all available algorithms (lazy loading)."""
        if not cls._algorithms:  # Only load once
            try:
                from .algorithms.greedy import GreedyAlgorithm
                cls._algorithms['greedy'] = GreedyAlgorithm
            except ImportError:
                pass

            try:
                from .algorithms.greedy_daily import GreedyDailyAlgorithm
                cls._algorithms['greedy_daily'] = GreedyDailyAlgorithm
            except ImportError:
                pass

            try:
                from .algorithms.neural_net import NeuralNetAlgorithm
                cls._algorithms['neural_net'] = NeuralNetAlgorithm
                cls._algorithms['nn'] = NeuralNetAlgorithm  # alias
            except ImportError:
                pass

            try:
                from .algorithms.genetic import GeneticAlgorithm
                cls._algorithms['genetic'] = GeneticAlgorithm
                cls._algorithms['ga'] = GeneticAlgorithm  # alias
            except ImportError:
                pass
