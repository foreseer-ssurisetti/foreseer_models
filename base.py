"""
Base classes and data structures for foreseer_models library.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class StrategyResults:
    """Standardized outputs for any strategy."""
    strategy_name: str
    weights: np.ndarray
    oos_sharpe_daily: float
    oos_sharpe_annual: float
    allocations_best: pd.DataFrame
    oos_returns: pd.Series
    model_path: Optional[str] = None
    training_metrics: Optional[Dict[str, Any]] = None


class BaseAlgorithm(ABC):
    """Abstract base class for all algorithms."""

    def __init__(self, name: str) -> None:
        self.name: str = name

    def fit(self, train_data: Any) -> None:
        """Train/prepare the model on training data."""
        pass

    @abstractmethod
    def predict(self, data: Any) -> np.ndarray:
        """Produce portfolio weights/signals for the given data."""
        pass

    @abstractmethod
    def backtest(self, data: Any) -> StrategyResults:
        """Perform splitting/rolling evaluation and return StrategyResults."""
        pass

    @abstractmethod
    def save_model(self, path: str) -> str:
        """Persist model state/metadata to 'path'."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model state/metadata from 'path'."""
        pass