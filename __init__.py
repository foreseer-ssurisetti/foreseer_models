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
from typing import Optional, Union, Type, Dict
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


def list_saved_models(algorithm: str = None) -> Dict[str, list]:
    """List all saved models, optionally filtered by algorithm type."""
    import os

    # Get the foreseer_models directory
    try:
        import foreseer_models
        foreseer_dir = os.path.dirname(foreseer_models.__file__)
        models_base_dir = os.path.join(foreseer_dir, "models")
    except:
        return {}

    if not os.path.exists(models_base_dir):
        return {}

    saved_models = {}

    # List all algorithm directories
    for algo_dir in os.listdir(models_base_dir):
        algo_path = os.path.join(models_base_dir, algo_dir)
        if os.path.isdir(algo_path):
            if algorithm is None or algorithm == algo_dir:
                # List all .pkl files in this algorithm directory
                pkl_files = [f for f in os.listdir(algo_path) if f.endswith('.pkl')]
                saved_models[algo_dir] = pkl_files

    return saved_models


def load_saved_model(algorithm: str, model_name: str) -> BaseAlgorithm:
    """Load a previously saved model by algorithm type and model name."""
    # Create algorithm instance
    algo = AlgorithmFactory.create(algorithm)

    # Load the saved state
    algo.load_model(model_name)  # The load_model method now handles the path automatically

    return algo


# Create a shorthand alias that's callable
def run(algorithm: str,
        data: pd.DataFrame,
        mode: str = "backtest",
        save_path: Optional[str] = None,
        **kwargs):
    """Shorthand alias for foreseer_models function."""
    return foreseer_models(algorithm, data, mode, save_path, **kwargs)


# Export main components
__all__ = [
    'foreseer_models',
    'run',  # Add the shorthand
    'list_models',
    'register_model',
    'list_saved_models',
    'load_saved_model',
    'StrategyResults',
    'BaseAlgorithm'
]
