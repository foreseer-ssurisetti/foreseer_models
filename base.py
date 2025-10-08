"""
Base classes and data structures for foreseer_models library.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd


@dataclass
class PositionDetail:
    """Details for a single position in the portfolio."""
    strategy_id: int
    strategy_name: str
    units: int
    direction: str  # 'LONG' or 'SHORT'
    num_legs: Optional[int] = None
    leg_details: Optional[List[Dict]] = None

    def __str__(self):
        leg_info = f", {self.num_legs} legs" if self.num_legs else ""
        return f"ID {self.strategy_id:3d} ({self.strategy_name}): {self.units:+3d} units ({self.direction}{leg_info})"

    def print_with_legs(self):
        """Print position with detailed leg information."""
        print(f"  {self}")
        if self.leg_details:
            for leg in self.leg_details:
                print(f"    └─ {leg['description']}: {leg['size']:+.2f} ({leg['direction']})")


@dataclass
class AllocationSnapshot:
    """Snapshot of portfolio allocation at a specific time."""
    date: pd.Timestamp
    allocation_vector: np.ndarray
    positions: List[PositionDetail]
    long_units: int
    short_units: int
    gross_units: int
    net_units: int
    total_legs: Optional[int] = None
    long_legs: Optional[int] = None
    short_legs: Optional[int] = None
    train_sharpe: Optional[float] = None
    training_start: Optional[pd.Timestamp] = None
    training_end: Optional[pd.Timestamp] = None

    # Greeks (to be implemented)
    portfolio_gamma: Optional[float] = None
    portfolio_vega: Optional[float] = None
    portfolio_theta: Optional[float] = None
    portfolio_skew_vega: Optional[float] = None

    def summary(self):
        """Return a formatted summary string."""
        lines = [
            f"Allocation for {self.date.strftime('%Y-%m')}",
            f"  Long Units: {self.long_units}",
            f"  Short Units: {self.short_units}",
            f"  Net Units: {self.net_units:+d}",
            f"  Gross Units: {self.gross_units}",
            f"  Number of Positions: {len(self.positions)}"
        ]

        if self.total_legs is not None:
            lines.extend([
                f"  Total Legs: {self.total_legs}",
                f"  Long Legs: {self.long_legs}",
                f"  Short Legs: {self.short_legs}"
            ])

        if self.train_sharpe is not None:
            lines.append(f"  Training Sharpe: {self.train_sharpe:.3f}")

        # # Greeks (commented out until implemented)
        # if self.portfolio_gamma is not None:
        #     lines.append(f"  Portfolio Gamma: {self.portfolio_gamma:.2f}")
        # if self.portfolio_vega is not None:
        #     lines.append(f"  Portfolio Vega: {self.portfolio_vega:.2f}")
        # if self.portfolio_theta is not None:
        #     lines.append(f"  Portfolio Theta: {self.portfolio_theta:.2f}")

        return "\n".join(lines)

    def print_positions(self, show_legs: bool = False):
        """Print all positions, optionally with leg details."""
        print(self.summary())
        print("\n  Strategy Details:")
        for pos in self.positions:
            if show_legs:
                pos.print_with_legs()
            else:
                print(f"    {pos}")

    def get_position_by_id(self, strategy_id: int) -> Optional[PositionDetail]:
        """Get position details for a specific strategy ID."""
        for pos in self.positions:
            if pos.strategy_id == strategy_id:
                return pos
        return None


@dataclass
class ModelSpecification:
    """Model configuration and metadata."""
    algorithm_name: str
    lookback: int
    max_net: int
    max_gross: int
    n_outright: int

    # Date ranges
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    backtest_start: pd.Timestamp
    backtest_end: pd.Timestamp

    # Features and labels (to be implemented)
    features_used: Optional[List[str]] = None
    target_variable: str = "monthly_returns"

    # Additional parameters
    find_second_best: bool = True
    step_up_to: int = 5
    min_difference: int = 2

    def __str__(self):
        lines = [
            f"Model: {self.algorithm_name}",
            f"Lookback: {self.lookback} months",
            f"Max Net: {self.max_net}, Max Gross: {self.max_gross}",
            f"Training: {self.training_start.strftime('%Y-%m')} to {self.training_end.strftime('%Y-%m')}",
            f"Backtest: {self.backtest_start.strftime('%Y-%m')} to {self.backtest_end.strftime('%Y-%m')}"
        ]
        return "\n".join(lines)


@dataclass
class PerformanceMetrics:
    """Performance metrics for the strategy."""
    # Sharpe ratios
    oos_sharpe_monthly: float
    oos_sharpe_annual: float

    # Returns statistics
    mean_return: float
    std_return: float
    median_return: float

    # Drawdown metrics (to be implemented)
    max_drawdown: Optional[float] = None
    max_drawdown_duration: Optional[int] = None

    # Win/loss metrics
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None

    # Risk metrics (to be implemented)
    # sortino_ratio: Optional[float] = None
    # calmar_ratio: Optional[float] = None
    # information_ratio: Optional[float] = None

    # Transaction costs (to be implemented)
    # total_transaction_costs: Optional[float] = None
    # avg_monthly_turnover: Optional[float] = None

    def print_summary(self):
        """Print performance summary."""
        print("\nPerformance Metrics:")
        print(f"  Sharpe Ratio (Annual):  {self.oos_sharpe_annual:.3f}")
        print(f"  Mean Monthly Return:    {self.mean_return:.4f}")
        print(f"  Std Dev Monthly Return: {self.std_return:.4f}")
        print(f"  Median Monthly Return:  {self.median_return:.4f}")

        if self.win_rate is not None:
            print(f"  Win Rate:               {self.win_rate:.2%}")

        # if self.max_drawdown is not None:
        #     print(f"  Max Drawdown:           {self.max_drawdown:.2%}")
        # if self.total_transaction_costs is not None:
        #     print(f"  Total Transaction Costs: {self.total_transaction_costs:.4f}")


@dataclass
class GreeksPnLBreakdown:
    """PnL breakdown by Greeks (to be implemented)."""
    date: pd.Timestamp
    total_pnl: float

    # Greeks PnL components (commented out until implemented)
    # gamma_pnl: Optional[float] = None
    # vega_pnl: Optional[float] = None
    # theta_pnl: Optional[float] = None
    # rolldown_theta_pnl: Optional[float] = None
    # delta_pnl: Optional[float] = None
    # unexplained_pnl: Optional[float] = None


@dataclass
class StrategyResults:
    """Standardized outputs for any strategy."""

    # =========================================================================
    # MODEL SPECIFICATION (required field)
    # =========================================================================
    model_spec: ModelSpecification

    # =========================================================================
    # OPTIONAL FIELDS (all have defaults)
    # =========================================================================

    # Strategy information
    strategy_legs_info: Optional[Dict[int, Dict]] = None

    # Performance metrics
    performance_metrics: Optional[PerformanceMetrics] = None
    performance_metrics_second: Optional[PerformanceMetrics] = None

    # PnL Time Series
    oos_returns: Optional[pd.Series] = None                    # Best allocation PnL
    oos_returns_second: Optional[pd.Series] = None             # Second-best PnL

    # Positions and Weights Time Series
    allocations_best: Optional[pd.DataFrame] = None            # Best allocation vectors over time
    allocations_second_best: Optional[pd.DataFrame] = None     # Second-best vectors
    allocation_snapshots: Optional[Dict[pd.Timestamp, AllocationSnapshot]] = None
    allocation_snapshots_second: Optional[Dict[pd.Timestamp, AllocationSnapshot]] = None

    # Tabular data
    results_table: Optional[pd.DataFrame] = None  # Comprehensive table with all info per date

    # Forward-looking predictions
    next_month_allocation: Optional[AllocationSnapshot] = None
    next_month_second_best: Optional[AllocationSnapshot] = None
    weights: Optional[np.ndarray] = None                       # Best allocation weights for next period
    weights_second_best: Optional[np.ndarray] = None           # Second-best weights

    # Comparison metrics
    sharpe_diff_history: Optional[pd.Series] = None

    # Model persistence
    model_path: Optional[str] = None

    # Greeks Time Series (to be implemented)
    # portfolio_gamma_series: Optional[pd.Series] = None
    # portfolio_vega_series: Optional[pd.Series] = None
    # portfolio_theta_series: Optional[pd.Series] = None

    # Transaction Costs Time Series (to be implemented)
    # transaction_costs_series: Optional[pd.Series] = None
    # turnover_series: Optional[pd.Series] = None

    # Greeks PnL Breakdown (to be implemented)
    # greeks_pnl_breakdown: Optional[List[GreeksPnLBreakdown]] = None

    # =========================================================================
    # METHODS FOR ACCESSING DATA
    # =========================================================================

    def print_summary(self):
        """Print high-level summary."""
        print("\n" + "="*70)
        print("MODEL SPECIFICATION")
        print("="*70)
        print(self.model_spec)

        if self.performance_metrics:
            print("\n" + "="*70)
            print("PERFORMANCE SUMMARY")
            print("="*70)
            self.performance_metrics.print_summary()

            if self.performance_metrics_second:
                print("\nSecond-Best Performance:")
                print(f"  Sharpe Ratio (Annual):  {self.performance_metrics_second.oos_sharpe_annual:.3f}")
                diff = self.performance_metrics.oos_sharpe_annual - self.performance_metrics_second.oos_sharpe_annual
                print(f"  Difference from Best:   {diff:+.3f}")

    def print_next_month(self, show_legs: bool = False, show_second_best: bool = True):
        """Print forward-looking allocation."""
        if not self.next_month_allocation:
            print("No forward-looking allocation available.")
            return

        print(f"\n{'=' * 70}")
        print(f"FORWARD-LOOKING ALLOCATION FOR {self.next_month_allocation.date.strftime('%Y-%m')}")
        print(f"{'=' * 70}")

        if self.next_month_allocation.training_start:
            print(f"Training period: {self.next_month_allocation.training_start.strftime('%Y-%m')} "
                  f"to {self.next_month_allocation.training_end.strftime('%Y-%m')}")

        print("\nBEST ALLOCATION:")
        self.next_month_allocation.print_positions(show_legs=show_legs)

        if show_second_best and self.next_month_second_best:
            print("\nSECOND-BEST ALLOCATION:")
            self.next_month_second_best.print_positions(show_legs=show_legs)

            # Compare allocations
            n_different = np.sum(
                self.next_month_allocation.allocation_vector !=
                self.next_month_second_best.allocation_vector
            )
            sharpe_diff = (self.next_month_allocation.train_sharpe -
                          self.next_month_second_best.train_sharpe)

            print(f"\n  Allocation Comparison:")
            print(f"    Strategies with Different Positions: {n_different}")
            print(f"    Sharpe Difference: {sharpe_diff:.3f}")

    def get_allocation_at_date(self, date: pd.Timestamp,
                               use_second_best: bool = False) -> Optional[AllocationSnapshot]:
        """Get allocation snapshot at a specific date."""
        snapshots = self.allocation_snapshots_second if use_second_best else self.allocation_snapshots
        if snapshots:
            return snapshots.get(date)
        return None

    def list_dates(self) -> List[pd.Timestamp]:
        """List all dates in the backtest."""
        if self.allocation_snapshots:
            return list(self.allocation_snapshots.keys())
        return []

    def get_results_at_date(self, date: pd.Timestamp) -> Optional[pd.Series]:
        """Get all results for a specific date from results_table."""
        if self.results_table is not None and date in self.results_table.index:
            return self.results_table.loc[date]
        return None

    # =========================================================================
    # UPDATE METHODS (for re-running analysis with new data)
    # =========================================================================

    def update_with_new_data(self, new_returns: pd.Series):
        """
        Update predictions with new data point (to be implemented).
        This will allow re-running the full analysis with updated data.
        """
        # TODO: Implement incremental update
        # - Append new returns to existing data
        # - Re-run optimization for next period
        # - Update predictions and allocations
        pass

    def add_performance_metric(self, metric_name: str, metric_value: float):
        """
        Add custom performance metric (to be implemented).
        Allows extending with new metrics without changing core structure.
        """
        # TODO: Implement dynamic metric addition
        # - Store in a flexible dict structure
        # - Allow for custom calculations
        pass

    # =========================================================================
    # ENSEMBLE METHODS (placeholder for future implementation)
    # =========================================================================

    def ensemble_with(self, other_results: 'StrategyResults', weight: float = 0.5):
        """
        Ensemble this strategy with another (to be implemented).

        Parameters:
        -----------
        other_results : StrategyResults
            Another strategy to ensemble with
        weight : float
            Weight for this strategy (1-weight for other)
        """
        # TODO: Implement ensembling
        # - Combine allocations from multiple strategies
        # - Weight by performance or custom weights
        # - Generate combined predictions
        pass


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