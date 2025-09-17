# =============================================================================
# File: foreseer_models/algorithms/greedy_daily.py
# =============================================================================
"""
Greedy Daily algorithm implementation.
Uses daily data with daily Sharpe calculation and rolling monthly windows.
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Any
from ..base import BaseAlgorithm, StrategyResults


class GreedyDailyAlgorithm(BaseAlgorithm):
    """
    Greedy Daily algorithm - uses daily data with daily Sharpe optimization.
    Rolling window is still monthly but operates on daily frequency.
    """

    def __init__(self, name: str = "GreedyDailyAlgorithm",
                 gross_cap: int = 20,
                 umax: int = 3,
                 step_up_to: int = 3,
                 net_strict: bool = False,
                 lookback_days: int = 30,  # Monthly window in days
                 rebalance_freq: int = 5):  # Rebalance every N days (weekly)
        super().__init__(name)
        self.gross_cap = gross_cap
        self.umax = umax
        self.step_up_to = step_up_to
        self.net_strict = net_strict
        self.lookback_days = lookback_days
        self.rebalance_freq = rebalance_freq
        self.fitted_params = {}
        self.last_allocation = None

    def _negative_sharpe_daily(self, X_daily: pd.DataFrame, u: np.ndarray) -> float:
        """
        Return the *negative* daily Sharpe for a portfolio with integer units u.

        X_daily : (T x N) DataFrame of daily returns over the training window.
        u       : (N,) integer units vector (long>0, short<0).

        We compute portfolio daily returns r = X_daily @ u (length T),
        then daily Sharpe = mean(r) / std(r). Return -Sharpe for minimizers.
        """
        if len(X_daily) < 2:  # Need at least 2 observations
            return 0.0

        pnl = X_daily.values @ u
        mu = pnl.mean()
        sd = pnl.std(ddof=1)
        sharpe = 0.0 if sd == 0 else mu / sd
        return -sharpe

    def _greedy_integer_alloc_daily(self, X_daily: pd.DataFrame) -> np.ndarray:
        """
        Greedy search on integer units using daily data and daily Sharpe.
        Same logic as monthly version but optimizes daily Sharpe.
        """
        N = X_daily.shape[1]
        u = np.zeros(N, dtype=int)

        # Current objective value (negative daily Sharpe)
        base = self._negative_sharpe_daily(X_daily, u)

        iteration = 0
        max_iterations = 1000

        while iteration < max_iterations:
            iteration += 1
            best = (0.0, None, 0)
            gross_used = np.abs(u).sum()

            for i in range(N):
                # remaining per-name headroom (|u_i| ≤ umax)
                head_name = self.umax - abs(u[i])
                if head_name <= 0:
                    continue

                # can't add more units than step_up_to, name headroom, or remaining gross capacity
                max_step_here = min(self.step_up_to, head_name, self.gross_cap - gross_used)
                if max_step_here <= 0:
                    continue

                for sgn in (+1, -1):
                    for step in range(1, max_step_here + 1):
                        v = u.copy()
                        v[i] += sgn * step

                        # Enforce gross cap early (skip infeasible)
                        if np.abs(v).sum() > self.gross_cap:
                            break

                        # Enforce net-long (≥0 or >0)
                        net_sum = v.sum()
                        net_ok = (net_sum > 0) if self.net_strict else (net_sum >= 0)
                        if not net_ok:
                            continue

                        # Evaluate negative daily Sharpe; lower is better
                        val = self._negative_sharpe_daily(X_daily, v)
                        reduction = base - val  # how much we *reduced* negative Sharpe
                        if reduction > best[0] + 1e-12:
                            best = (reduction, i, sgn * step)

            # If no move improved the objective, stop
            if best[1] is None:
                break

            # Apply the best move
            u[best[1]] += best[2]
            # Update base (we reduced negative Sharpe by 'best[0]')
            base -= best[0]

            # Stop early if gross cap hit
            if np.abs(u).sum() >= self.gross_cap:
                break

        # Local polish: try ±1 on each name; accept if improves
        improved = True
        polish_iterations = 0
        max_polish = 100

        while improved and polish_iterations < max_polish:
            polish_iterations += 1
            improved = False
            base = self._negative_sharpe_daily(X_daily, u)

            for i in range(N):
                for sgn in (+1, -1):
                    # obey per-name cap
                    if abs(u[i] + sgn) > self.umax:
                        continue

                    v = u.copy()
                    v[i] += sgn

                    # obey gross cap
                    if np.abs(v).sum() > self.gross_cap:
                        continue

                    # obey net-long
                    net_sum = v.sum()
                    net_ok = (net_sum > 0) if self.net_strict else (net_sum >= 0)
                    if not net_ok:
                        continue

                    val = self._negative_sharpe_daily(X_daily, v)
                    if val < base - 1e-12:  # lower negative Sharpe = better
                        u = v
                        base = val
                        improved = True
                        break
                if improved:
                    break

        return u

    def fit(self, train_data: pd.DataFrame) -> None:
        """Store relevant statistics and validate data format."""
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("Training data must be a pandas DataFrame")

        self.fitted_params = {
            'n_assets': train_data.shape[1],
            'asset_names': list(train_data.columns),
            'training_period': (train_data.index[0], train_data.index[-1]),
            'gross_cap': self.gross_cap,
            'umax': self.umax,
            'step_up_to': self.step_up_to,
            'net_strict': self.net_strict,
            'lookback_days': self.lookback_days,
            'rebalance_freq': self.rebalance_freq
        }

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Produce portfolio weights for the given daily training window."""
        if not hasattr(self, 'fitted_params') or not self.fitted_params:
            raise ValueError("Algorithm must be fitted before prediction")

        # Use the last lookback_days for training
        if len(data) > self.lookback_days:
            train_data = data.tail(self.lookback_days)
        else:
            train_data = data

        units = self._greedy_integer_alloc_daily(train_data)
        self.last_allocation = units

        # Convert to weights (units / total_gross_units)
        total_gross = np.abs(units).sum()
        if total_gross == 0:
            weights = np.zeros(len(units))
        else:
            weights = units / total_gross

        return weights

    def backtest(self, returns_d: pd.DataFrame) -> StrategyResults:
        """
        Perform rolling daily backtest using greedy daily algorithm.
        Rebalances every rebalance_freq days using lookback_days of training data.
        """
        if len(returns_d) < self.lookback_days + self.rebalance_freq:
            raise ValueError(f"Insufficient data: need at least {self.lookback_days + self.rebalance_freq} days")

        results = []
        allocs = {}

        # Start after we have enough lookback data
        start_idx = self.lookback_days

        for t in range(start_idx, len(returns_d), self.rebalance_freq):
            # Get training window (lookback_days of data ending at t-1)
            train_start = max(0, t - self.lookback_days)
            train_data = returns_d.iloc[train_start:t]

            # Fit and get allocation
            self.fit(train_data)
            u = self._greedy_integer_alloc_daily(train_data)

            # Hold period: from t to min(t + rebalance_freq, end)
            hold_end = min(t + self.rebalance_freq, len(returns_d))
            hold_period = returns_d.iloc[t:hold_end]

            if len(hold_period) == 0:
                break

            # Calculate PnL for the hold period
            hold_pnl = (hold_period.values @ u).sum()  # Sum of daily PnLs

            # Calculate training metrics
            train_sharpe_daily = -self._negative_sharpe_daily(train_data, u)

            # Store results for each day in hold period (or just the period)
            period_date = returns_d.index[t]
            allocs[period_date] = u

            results.append({
                "date": period_date,
                "period_pnl": hold_pnl,
                "daily_pnl": hold_pnl / len(hold_period),  # Average daily PnL
                "sh_train_daily": train_sharpe_daily,
                "gross_units": int(np.abs(u).sum()),
                "net_units": int(u.sum()),
                "hold_days": len(hold_period)
            })

        res = pd.DataFrame(results).set_index("date")

        if len(res) == 0:
            return StrategyResults(
                strategy_name=self.name,
                weights=np.array([]),
                oos_sharpe_daily=0.0,
                oos_sharpe_annual=0.0,
                allocations_best=pd.DataFrame(),
                oos_returns=pd.Series(dtype=float),
                training_metrics={"no_data": True}
            )

        # Calculate out-of-sample daily Sharpe
        daily_pnls = res["daily_pnl"]
        mu_oos = daily_pnls.mean()
        sd_oos = daily_pnls.std(ddof=1)
        sharpe_oos_daily = 0.0 if sd_oos == 0 else mu_oos / sd_oos

        # Get final allocation weights
        if len(allocs) > 0:
            last_date = list(allocs.keys())[-1]
            last_allocation = allocs[last_date]
            total_gross = np.abs(last_allocation).sum()
            final_weights = last_allocation / total_gross if total_gross > 0 else last_allocation
        else:
            final_weights = np.array([])

        # Create allocations DataFrame
        if len(returns_d.columns) == len(final_weights):
            allocations_df = pd.DataFrame({
                'strat': list(returns_d.columns),
                'weight': final_weights
            })
        else:
            allocations_df = pd.DataFrame()

        return StrategyResults(
            strategy_name=self.name,
            weights=final_weights,
            oos_sharpe_daily=sharpe_oos_daily,
            oos_sharpe_annual=sharpe_oos_daily * np.sqrt(252),  # Annualize daily Sharpe
            allocations_best=allocations_df,
            oos_returns=daily_pnls,  # Daily PnL series
            training_metrics={
                "allocations_by_date": allocs,
                "backtest_results": res,
                "total_periods": len(res),
                "avg_gross_units": res["gross_units"].mean(),
                "avg_net_units": res["net_units"].mean(),
                "avg_train_sharpe_daily": res["sh_train_daily"].mean(),
                "avg_hold_days": res["hold_days"].mean(),
                "lookback_days": self.lookback_days,
                "rebalance_freq": self.rebalance_freq
            }
        )

    def save_model(self, path: str) -> str:
        """Save the greedy daily algorithm parameters and fitted state."""
        # Get the foreseer_models directory
        import os
        import foreseer_models

        # Create full path inside foreseer_models/models/
        foreseer_dir = os.path.dirname(foreseer_models.__file__)
        models_dir = os.path.join(foreseer_dir, "models", "greedy_daily")

        # Ensure path doesn't have directory separators (security)
        filename = os.path.basename(path)
        if not filename.endswith('.pkl'):
            filename = f"{filename}.pkl"

        full_path = os.path.join(models_dir, filename)

        model_state = {
            'name': self.name,
            'gross_cap': self.gross_cap,
            'umax': self.umax,
            'step_up_to': self.step_up_to,
            'net_strict': self.net_strict,
            'lookback_days': self.lookback_days,
            'rebalance_freq': self.rebalance_freq,
            'fitted_params': self.fitted_params,
            'last_allocation': self.last_allocation
        }

        # Create directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        with open(full_path, 'wb') as f:
            pickle.dump(model_state, f)

        return full_path

    def load_model(self, path: str) -> None:
        """Load the greedy daily algorithm parameters and fitted state."""
        # Handle both full paths and just filenames
        if not os.path.isabs(path) and not os.path.exists(path):
            # Try to find it in the models directory
            import foreseer_models
            foreseer_dir = os.path.dirname(foreseer_models.__file__)
            models_path = os.path.join(foreseer_dir, "models", "greedy_daily", path)
            if not path.endswith('.pkl'):
                models_path = f"{models_path}.pkl"
            if os.path.exists(models_path):
                path = models_path

        with open(path, 'rb') as f:
            model_state = pickle.load(f)

        self.name = model_state['name']
        self.gross_cap = model_state['gross_cap']
        self.umax = model_state['umax']
        self.step_up_to = model_state['step_up_to']
        self.net_strict = model_state['net_strict']
        self.lookback_days = model_state['lookback_days']
        self.rebalance_freq = model_state['rebalance_freq']
        self.fitted_params = model_state['fitted_params']
        self.last_allocation = model_state['last_allocation']