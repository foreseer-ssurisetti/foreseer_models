"""
Foreseer Models Library
A factory-based library for portfolio optimization algorithms.

Usage:
    # Option 1: Import the function directly
    from foreseer_models import foreseer_models
    results = foreseer_models("greedy", data=returns_df, gross_cap=20)

    # Option 2: Import and call module function
    import foreseer_models
    results = foreseer_models.foreseer_models("greedy", data=returns_df)

    # Option 3: Use the shorthand (recommended)
    import foreseer_models as fm
    results = fm("greedy", data=returns_df)
"""

from .factory import AlgorithmFactory
from .base import StrategyResults, BaseAlgorithm
from typing import Optional, Union, Type
import pandas as pd
import numpy as np


def foreseer_models(algorithm: str,
                   data: pd.DataFrame,
                   mode: str = "backtest",
                   save_path: Optional[str] = None,
                   **kwargs) -> Union[StrategyResults, BaseAlgorithm, np.ndarray]:
    """
    Main entry point for the foreseer_models library.

    Args:
        algorithm: Algorithm name ('greedy', 'neural_net', 'genetic')
        data: Input data (DataFrame of returns)
        mode: Operation mode ('backtest', 'predict', 'create')
        save_path: Optional path to save the model
        **kwargs: Algorithm-specific parameters

    Returns:
        StrategyResults: If mode='backtest'
        np.ndarray: If mode='predict'
        BaseAlgorithm: If mode='create'
    """

    # Create algorithm instance
    algo = AlgorithmFactory.create(algorithm, **kwargs)

    if mode == "create":
        return algo

    elif mode == "predict":
        algo.fit(data)
        weights = algo.predict(data)

        if save_path:
            algo.save_model(save_path)

        return weights

    elif mode == "backtest":
        results = algo.backtest(data)

        if save_path:
            model_path = algo.save_model(save_path)
            results.model_path = model_path

        return results

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'backtest', 'predict', or 'create'")


def list_models() -> list:
    """List all available algorithms."""
    return AlgorithmFactory.list_algorithms()


def register_model(name: str, algorithm_class: Type[BaseAlgorithm]) -> None:
    """Register a custom algorithm."""
    AlgorithmFactory.register(name, algorithm_class)


# Create a shorthand alias that's callable
def run(algorithm: str,
        data: pd.DataFrame,
        mode: str = "backtest",
        save_path: Optional[str] = None,
        **kwargs):
    """Shorthand alias for foreseer_models function."""
    return foreseer_models(algorithm, data, mode, save_path, **kwargs)#


# Export main components
__all__ = [
    'foreseer_models',
    'list_models',
    'register_model',
    'StrategyResults',
    'BaseAlgorithm'
]
