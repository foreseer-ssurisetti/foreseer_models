# Foreseer Models Library

A factory-based library for portfolio optimization algorithms with comprehensive tracking of allocations, positions, and performance metrics.

## Directory Structure

```
foreseer_models/
├── __init__.py              # Main entry point
├── base.py                  # Base classes and data structures
├── factory.py               # Algorithm factory
├── models/                  # Auto-created saved models
│   ├── greedy_with_legs/
│   │   ├── production_v1.pkl
│   │   └── experiment_1.pkl
│   └── neural_net/
└── algorithms/              # Algorithm implementations
    ├── greedy.py
    ├── greedy_with_legs.py  # Main algorithm with leg tracking
    └── neural_net.py
```

## Core Features

### What's Available Now

✅ **Model Specification**
- Algorithm configuration and parameters
- Date ranges (training and backtest periods)
- Features and labels tracking

✅ **Time Series Data**
- PnL time series (best and second-best)
- Position and weight time series
- Allocation snapshots at each period

✅ **Tabular Results**
- Comprehensive results table with all metrics per date
- Strategy details and positions
- Performance measures

✅ **Forward-Looking Predictions**
- Next period allocation (best and second-best)
- Strategy positions with leg details
- Performance comparison

✅ **Model Persistence**
- Save and load trained models
- Maintain full state and configuration

### Coming Soon

🔄 **Greeks PnL Breakdown** (commented out in code)
- Gamma PnL contribution
- Vega PnL contribution  
- Theta PnL contribution
- Rolldown Theta PnL

🔄 **Transaction Costs** (commented out in code)
- Per-period transaction costs
- Turnover metrics
- Net PnL after costs

🔄 **Update Capabilities**
- Add new data points incrementally
- Re-run full analysis on updated data

🔄 **Performance Measures**
- Additional risk metrics (Sortino, Calmar)
- Drawdown analysis
- Custom metric support

🔄 **Ensemble Methods**
- Combine multiple strategies
- Weighted averaging
- Meta-model predictions

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from foreseer_models import foreseer_models
import pandas as pd

# Load returns data
returns = pd.read_excel("Cube.xlsx", sheet_name='SPX_BlackPL_Piv', 
                       index_col=0, parse_dates=True) / 1e7

# Run backtest
results = foreseer_models(
    "greedy_with_legs",
    data=returns,
    lookback=24,
    max_net=6,
    max_gross=10,
    strat_legs_file="strat.xlsx",
    save_path="production_v1"
)

# View results
results.print_summary()
results.print_next_month(show_legs=True)
```

## Data Structures

### StrategyResults

The main results object containing all backtest outputs:

```python
results = StrategyResults(
    # Model Specification
    model_spec: ModelSpecification       # Algorithm config and date ranges
    strategy_legs_info: Dict             # Strategy leg information
    
    # Performance Metrics
    performance_metrics: PerformanceMetrics        # Best allocation metrics
    performance_metrics_second: PerformanceMetrics # Second-best metrics
    
    # Time Series Data
    oos_returns: pd.Series               # PnL time series (best)
    oos_returns_second: pd.Series        # PnL time series (second-best)
    allocations_best: pd.DataFrame       # Weight vectors over time
    allocation_snapshots: Dict           # Detailed snapshots per date
    
    # Tabular Data
    results_table: pd.DataFrame          # Comprehensive results per date
    
    # Forward-Looking
    next_month_allocation: AllocationSnapshot  # Next period prediction
    weights: np.ndarray                        # Next period weights
)
```

### AllocationSnapshot

Portfolio state at a specific time:

```python
snapshot = AllocationSnapshot(
    date: pd.Timestamp
    allocation_vector: np.ndarray
    positions: List[PositionDetail]
    long_units, short_units, net_units, gross_units: int
    total_legs, long_legs, short_legs: int  # If legs file provided
    train_sharpe: float
    
    # Coming soon:
    # portfolio_gamma, portfolio_vega, portfolio_theta: float
)
```

### ModelSpecification

Model configuration and metadata:

```python
spec = ModelSpecification(
    algorithm_name: str
    lookback: int
    max_net, max_gross, n_outright: int
    training_start, training_end: pd.Timestamp
    backtest_start, backtest_end: pd.Timestamp
)
```

### PerformanceMetrics

Performance statistics:

```python
metrics = PerformanceMetrics(
    oos_sharpe_monthly, oos_sharpe_annual: float
    mean_return, std_return, median_return: float
    win_rate, avg_win, avg_loss: float
    
    # Coming soon:
    # max_drawdown, sortino_ratio, calmar_ratio: float
    # total_transaction_costs, avg_monthly_turnover: float
)
```

## Usage Examples

### 1. Basic Backtest

```python
results = foreseer_models(
    "greedy_with_legs",
    data=returns,
    mode="backtest",
    lookback=24,
    find_second_best=True,
    strat_legs_file="strat.xlsx"
)

# Access model specification
print(results.model_spec)

# Access performance
results.performance_metrics.print_summary()
```

### 2. View Time Series Data

```python
# PnL time series
pnl = results.oos_returns
print(pnl.head())

# Plot PnL
import matplotlib.pyplot as plt
pnl.cumsum().plot()
plt.title('Cumulative PnL')
plt.show()

# Allocations over time
allocations = results.allocations_best
print(allocations.iloc[:5, :5])  # First 5 periods, first 5 strategies
```

### 3. Access Tabular Results

```python
# Full results table
table = results.results_table
print(table.head())

# Columns: date, pnl, train_sharpe, long_units, short_units, 
#          net_units, gross_units, num_positions, total_legs, etc.

# Filter by criteria
high_sharpe = table[table['train_sharpe'] > 1.5]
print(high_sharpe)

# Get specific date
date_results = results.get_results_at_date(pd.Timestamp('2024-01-31'))
print(date_results)
```

### 4. Access Allocation Snapshots

```python
# List all dates
dates = results.list_dates()

# Get snapshot for specific date
allocation = results.get_allocation_at_date(dates[0])
allocation.print_positions(show_legs=True)

# Access snapshot properties
print(f"Net: {allocation.net_units}")
print(f"Gross: {allocation.gross_units}")
print(f"Positions: {len(allocation.positions)}")

# Iterate through positions
for pos in allocation.positions:
    print(f"Strategy {pos.strategy_id}: {pos.units} units")
    if pos.leg_details:
        for leg in pos.leg_details:
            print(f"  {leg['description']}: {leg['size']}")
```

### 5. Compare Best vs Second-Best

```python
if results.weights_second_best is not None:
    # Performance comparison
    best = results.performance_metrics
    second = results.performance_metrics_second
    print(f"Best Sharpe: {best.oos_sharpe_annual:.3f}")
    print(f"Second Sharpe: {second.oos_sharpe_annual:.3f}")
    
    # Position comparison
    best_alloc = results.next_month_allocation
    second_alloc = results.next_month_second_best
    
    best_ids = {pos.strategy_id for pos in best_alloc.positions}
    second_ids = {pos.strategy_id for pos in second_alloc.positions}
    print(f"Common: {len(best_ids & second_ids)}")
    
    # Sharpe difference over time
    sharpe_diff = results.sharpe_diff_history
    print(sharpe_diff.describe())
```

### 6. Forward-Looking Predictions

```python
# Print next month allocation
results.print_next_month(show_legs=True, show_second_best=True)

# Access programmatically
next_alloc = results.next_month_allocation
print(f"Date: {next_alloc.date}")
print(f"Net: {next_alloc.net_units}")
print(f"Training Sharpe: {next_alloc.train_sharpe:.3f}")

# Get weights vector
weights = results.weights  # np.ndarray
print(f"Non-zero positions: {np.sum(weights != 0)}")
```

### 7. Save and Load Models

```python
# Save during backtest
results = foreseer_models(
    "greedy_with_legs",
    data=returns,
    save_path="production_v1"
)

# List saved models
from foreseer_models import list_saved_models
models = list_saved_models("greedy_with_legs")

# Load saved model
from foreseer_models import load_saved_model
algo = load_saved_model("greedy_with_legs", "production_v1")

# Use loaded model
new_weights = algo.predict(returns)
new_results = algo.backtest(returns)
```

### 8. Export Data

```python
# Export results table
results.results_table.to_excel("backtest_results.xlsx")

# Export allocations
results.allocations_best.to_csv("allocations.csv")

# Export PnL
results.oos_returns.to_csv("pnl.csv")
```

## Algorithm Parameters

### greedy_with_legs

```python
{
    'lookback': 24,           # Training window (months)
    'max_net': 6,             # Max net position
    'max_gross': 10,          # Max gross position
    'n_outright': 90,         # Strategies that can be shorted
    'step_up_to': 5,          # Max step size in greedy search
    'find_second_best': True, # Find second-best allocation
    'min_difference': 2,      # Min positions difference for second-best
    'strat_legs_file': None   # Path to strategy legs Excel file
}
```

## Strategy Legs File Format

The `strat.xlsx` file should have columns:

| ID | L1      | L1_Siz | L2      | L2_Siz | L3      | L3_Siz | L4      | L4_Siz |
|----|---------|--------|---------|--------|---------|--------|---------|--------|
| 1  | 1wkd05P | 1.0    | 1wkd05C | -1.0   |         |        |         |        |
| 2  | 2Md25C  | 2.0    | 2Md25P  | -2.0   | 2Md10P  | 1.0    |         |        |

## Coming Soon Features

### Greeks PnL Breakdown

```python
# When implemented:
greeks = results.greeks_pnl_breakdown
for breakdown in greeks:
    print(f"{breakdown.date}")
    print(f"  Gamma PnL: {breakdown.gamma_pnl}")
    print(f"  Vega PnL: {breakdown.vega_pnl}")
    print(f"  Theta PnL: {breakdown.theta_pnl}")
```

### Transaction Costs

```python
# When implemented:
costs = results.transaction_costs_series
print(f"Total costs: {results.performance_metrics.total_transaction_costs}")
print(f"Avg turnover: {results.performance_metrics.avg_monthly_turnover}")
```

### Update with New Data

```python
# When implemented:
new_return = pd.Series([0.02], index=[pd.Timestamp('2025-11-30')])
results.update_with_new_data(new_return)
```

### Ensemble Methods

```python
# When implemented:
results1 = foreseer_models("greedy_with_legs", data=returns)
results2 = foreseer_models("neural_net", data=returns)
ensemble = results1.ensemble_with(results2, weight=0.6)
```
