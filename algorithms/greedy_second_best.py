"""
Greedy algorithm with second-best allocation tracking.
Enhanced version that finds both optimal and second-best solutions.
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Optional, Dict, Any
from ..base import BaseAlgorithm, StrategyResults


class GreedySecondBestAlgorithm(BaseAlgorithm):
    """
    Greedy integer allocation optimizer with second-best solution tracking.

    Finds both the optimal allocation and a substantially different second-best
    allocation for robustness analysis and alternative portfolio construction.
    """

    def __init__(self,
                 lookback: int = 24,
                 max_long: int = 5,
                 max_short: int = 5,
                 umax: int = 3,
                 step_up_to: int = 5,
                 net_strict: bool = False,
                 find_second_best: bool = True,
                 min_difference: int = 2,
                 scale_factor: float = 1e7):
        """
        Parameters:
        -----------
        lookback : int
            Number of months for training window
        max_long : int
            Maximum total long units
        max_short : int
            Maximum total short units
        umax : int
            Maximum units per strategy
        step_up_to : int
            Maximum step size in greedy search
        net_strict : bool
            If True, net position must be > 0 (not just >= 0)
        find_second_best : bool
            Whether to find second-best allocation
        min_difference : int
            Minimum number of strategies that must differ between best and second-best
        scale_factor : float
            Scaling factor for returns (default 1e7)
        """
        super().__init__(name="greedy_second_best")

        self.lookback = lookback
        self.max_long = max_long
        self.max_short = max_short
        self.umax = umax
        self.step_up_to = step_up_to
        self.net_strict = net_strict
        self.find_second_best = find_second_best
        self.min_difference = min_difference
        self.scale_factor = scale_factor

        # Store results
        self.last_weights = None
        self.last_weights_second = None
        self.allocations_history = {}
        self.second_best_history = {}

    def _to_monthly_blocks(self, returns_d: pd.DataFrame,
                           drop_last_partial: bool = True) -> pd.DataFrame:
        """Convert daily returns to calendar monthly blocks."""
        if not isinstance(returns_d.index, pd.DatetimeIndex):
            returns_d.index = pd.to_datetime(returns_d.index)

        monthly = returns_d.groupby(pd.Grouper(freq='ME'))
        m_ret = monthly.apply(lambda x: (1 + x).prod() - 1)

        if drop_last_partial and len(returns_d) > 0:
            last_date = returns_d.index[-1]
            month_end = last_date + pd.offsets.MonthEnd(0)
            if last_date < month_end:
                m_ret = m_ret.iloc[:-1]

        m_ret.index.name = "date"
        return m_ret

    def _negative_sharpe(self, Xm: pd.DataFrame, u: np.ndarray) -> float:
        """Return negative monthly Sharpe for portfolio with units u."""
        pnl = Xm.values @ u
        mu = pnl.mean()
        sd = pnl.std(ddof=1)

        if sd == 0 or np.isnan(sd):
            return 0.0

        sharpe = mu / sd
        return -sharpe

    def _greedy_integer_alloc(self,
                              Xm: pd.DataFrame,
                              exclude_allocation: np.ndarray = None) -> np.ndarray:
        """
        Greedy search for optimal integer allocation.

        Parameters:
        -----------
        exclude_allocation : np.ndarray, optional
            If provided, find allocation that differs by at least min_difference positions
        """
        N = Xm.shape[1]
        u = np.zeros(N, dtype=int)

        base = self._negative_sharpe(Xm, u)
        iteration = 0
        max_iterations = 2500

        # Main greedy loop
        while iteration < max_iterations:
            iteration += 1
            best = (0.0, None, 0)

            long_used = u[u > 0].sum()
            short_used = np.abs(u[u < 0]).sum()

            for i in range(N):
                head_name = self.umax - abs(u[i])
                if head_name <= 0:
                    continue

                for sgn in (+1, -1):
                    if sgn > 0:
                        remaining = self.max_long - long_used
                    else:
                        remaining = self.max_short - short_used

                    if remaining <= 0:
                        continue

                    max_step = min(self.step_up_to, head_name, remaining)
                    if max_step <= 0:
                        continue

                    for step in range(1, max_step + 1):
                        v = u.copy()
                        v[i] += sgn * step

                        new_long = v[v > 0].sum()
                        new_short = np.abs(v[v < 0]).sum()

                        if new_long > self.max_long or new_short > self.max_short:
                            break

                        if new_long + new_short > (self.max_long + self.max_short):
                            break

                        net_sum = v.sum()
                        if self.net_strict:
                            if net_sum <= 0:
                                continue
                        else:
                            if net_sum < 0:
                                continue

                        # Check exclusion constraint
                        if exclude_allocation is not None:
                            n_different = np.sum(v != exclude_allocation)
                            if n_different < self.min_difference:
                                continue

                        val = self._negative_sharpe(Xm, v)
                        improvement = base - val

                        if improvement > best[0] + 1e-12:
                            best = (improvement, i, sgn * step)

            if best[1] is None:
                break

            u[best[1]] += best[2]
            base -= best[0]

            long_used = u[u > 0].sum()
            short_used = np.abs(u[u < 0]).sum()
            if long_used >= self.max_long and short_used >= self.max_short:
                break

        # Local polish
        improved = True
        polish_iter = 0
        max_polish = 100

        while improved and polish_iter < max_polish:
            polish_iter += 1
            improved = False
            base = self._negative_sharpe(Xm, u)

            for i in range(N):
                for sgn in (+1, -1):
                    if abs(u[i] + sgn) > self.umax:
                        continue

                    v = u.copy()
                    v[i] += sgn

                    new_long = v[v > 0].sum()
                    new_short = np.abs(v[v < 0]).sum()

                    if new_long > self.max_long or new_short > self.max_short:
                        continue

                    net_sum = v.sum()
                    if self.net_strict:
                        if net_sum <= 0:
                            continue
                    else:
                        if net_sum < 0:
                            continue

                    # Check exclusion constraint
                    if exclude_allocation is not None:
                        n_different = np.sum(v != exclude_allocation)
                        if n_different < self.min_difference:
                            continue

                    val = self._negative_sharpe(Xm, v)
                    if val < base - 1e-12:
                        u = v
                        base = val
                        improved = True
                        break

                if improved:
                    break

        return u

    def _find_top_k_allocations(self, Xm: pd.DataFrame, k: int = 2):
        """
        Find top k different allocations.

        Returns list of (negative_sharpe, allocation) tuples sorted by performance.
        """
        solutions = []

        for rank in range(k):
            if rank == 0:
                # Find best allocation
                u = self._greedy_integer_alloc(Xm, exclude_allocation=None)
            else:
                # Find next best allocation different from the best one
                u = self._greedy_integer_alloc(
                    Xm,
                    exclude_allocation=solutions[0][1]
                )

            score = self._negative_sharpe(Xm, u)
            solutions.append((score, u.copy()))

        return solutions

    def fit(self, train_data: pd.DataFrame) -> None:
        """Train the model on historical data."""
        # Convert to monthly if needed
        Xm = self._to_monthly_blocks(train_data, drop_last_partial=True)

        # Find best and second-best allocations
        if self.find_second_best:
            solutions = self._find_top_k_allocations(Xm, k=2)
            self.last_weights = solutions[0][1]
            self.last_weights_second = solutions[1][1]
        else:
            self.last_weights = self._greedy_integer_alloc(Xm)
            self.last_weights_second = None

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Produce portfolio weights for the given data.

        Returns the best allocation weights.
        """
        self.fit(data)
        return self.last_weights

    def backtest(self, data: pd.DataFrame) -> StrategyResults:
        """
        Perform walk-forward backtest.

        Returns:
        --------
        StrategyResults with both best and second-best performance metrics
        """
        # Scale returns if needed
        if self.scale_factor != 1.0:
            returns = data / self.scale_factor
        else:
            returns = data

        Xm = self._to_monthly_blocks(returns, drop_last_partial=True)
        dates = Xm.index

        if len(dates) <= self.lookback + 1:
            raise ValueError(
                f"Insufficient data: need at least {self.lookback + 2} months, "
                f"got {len(dates)} months"
            )

        results = []
        allocs_best = {}
        allocs_second = {}

        # Walk-forward backtest
        for k in range(self.lookback, len(dates) - 1):
            train = Xm.iloc[k - self.lookback:k]
            hold = Xm.iloc[k]

            # Find best and second-best allocations
            if self.find_second_best:
                solutions = self._find_top_k_allocations(train, k=2)
                u_best = solutions[0][1]
                u_second = solutions[1][1]

                pnl_hold_best = float(hold.values @ u_best)
                pnl_hold_second = float(hold.values @ u_second)
                train_sharpe_best = -solutions[0][0]
                train_sharpe_second = -solutions[1][0]
            else:
                u_best = self._greedy_integer_alloc(train)
                u_second = None
                pnl_hold_best = float(hold.values @ u_best)
                pnl_hold_second = None
                train_sharpe_best = -self._negative_sharpe(train, u_best)
                train_sharpe_second = None

            allocs_best[dates[k]] = u_best
            if self.find_second_best:
                allocs_second[dates[k]] = u_second

            long_units = int(u_best[u_best > 0].sum())
            short_units = int(np.abs(u_best[u_best < 0]).sum())

            result = {
                "date": dates[k],
                "pnl": pnl_hold_best,
                "sh_train": train_sharpe_best,
                "long_units": long_units,
                "short_units": short_units,
                "gross_units": long_units + short_units,
                "net_units": int(u_best.sum())
            }

            if self.find_second_best:
                result["pnl_second"] = pnl_hold_second
                result["sh_train_second"] = train_sharpe_second
                result["sharpe_diff"] = train_sharpe_best - train_sharpe_second

            results.append(result)

        res = pd.DataFrame(results).set_index("date")

        # Calculate OOS metrics
        mu_oos = res["pnl"].mean()
        sd_oos = res["pnl"].std(ddof=1)
        sharpe_oos_monthly = 0.0 if sd_oos == 0 else mu_oos / sd_oos
        sharpe_oos_annual = sharpe_oos_monthly * np.sqrt(12)

        if self.find_second_best:
            mu_oos_second = res["pnl_second"].mean()
            sd_oos_second = res["pnl_second"].std(ddof=1)
            sharpe_oos_second_monthly = 0.0 if sd_oos_second == 0 else mu_oos_second / sd_oos_second
            sharpe_oos_annual_second = sharpe_oos_second_monthly * np.sqrt(12)
        else:
            sharpe_oos_annual_second = None

        # Generate next month allocation
        train_final = Xm.iloc[-self.lookback:]

        if self.find_second_best:
            solutions = self._find_top_k_allocations(train_final, k=2)
            u_next = solutions[0][1]
            u_next_second = solutions[1][1]
            train_sharpe_next = -solutions[0][0]
            train_sharpe_next_second = -solutions[1][0]
        else:
            u_next = self._greedy_integer_alloc(train_final)
            u_next_second = None
            train_sharpe_next = -self._negative_sharpe(train_final, u_next)
            train_sharpe_next_second = None

        last_date = Xm.index[-1]
        next_month = last_date + pd.DateOffset(months=1)
        next_month = next_month + pd.offsets.MonthEnd(0)

        allocs_best[next_month] = u_next
        if self.find_second_best:
            allocs_second[next_month] = u_next_second

        # Format next month info
        positions_best = []
        for i, units in enumerate(u_next):
            if units != 0:
                positions_best.append({
                    'strategy': f'Strategy_{i + 1}',
                    'units': int(units),
                    'direction': 'LONG' if units > 0 else 'SHORT'
                })

        positions_second = []
        if self.find_second_best:
            for i, units in enumerate(u_next_second):
                if units != 0:
                    positions_second.append({
                        'strategy': f'Strategy_{i + 1}',
                        'units': int(units),
                        'direction': 'LONG' if units > 0 else 'SHORT'
                    })

        long_units = int(u_next[u_next > 0].sum())
        short_units = int(np.abs(u_next[u_next < 0]).sum())

        next_month_info = {
            'date': next_month,
            'training_start': train_final.index[0],
            'training_end': train_final.index[-1],
            'train_sharpe': train_sharpe_next,
            'train_sharpe_second': train_sharpe_next_second,
            'long_units': long_units,
            'short_units': short_units,
            'gross_units': long_units + short_units,
            'net_units': int(u_next.sum()),
            'positions_best': positions_best,
            'positions_second': positions_second,
            'allocation_second': u_next_second.tolist() if u_next_second is not None else None
        }

        # Store for later use
        self.allocations_history = allocs_best
        self.second_best_history = allocs_second
        self.last_weights = u_next
        self.last_weights_second = u_next_second

        # Convert allocations to DataFrame
        allocs_df = pd.DataFrame.from_dict(allocs_best, orient='index')
        allocs_df.index.name = 'date'

        allocs_second_df = None
        if self.find_second_best:
            allocs_second_df = pd.DataFrame.from_dict(allocs_second, orient='index')
            allocs_second_df.index.name = 'date'

        # Create StrategyResults
        return StrategyResults(
            strategy_name=self.name,
            weights=u_next,
            oos_sharpe_daily=sharpe_oos_monthly / np.sqrt(21),  # Approximate
            oos_sharpe_annual=sharpe_oos_annual,
            allocations_best=allocs_df,
            oos_returns=res["pnl"],
            weights_second_best=u_next_second,
            oos_sharpe_annual_second=sharpe_oos_annual_second,
            allocations_second_best=allocs_second_df,
            oos_returns_second=res["pnl_second"] if self.find_second_best else None,
            sharpe_diff_history=res["sharpe_diff"] if self.find_second_best else None,
            next_month_info=next_month_info,
            training_metrics={
                'lookback': self.lookback,
                'max_long': self.max_long,
                'max_short': self.max_short,
                'umax': self.umax,
                'find_second_best': self.find_second_best,
                'min_difference': self.min_difference
            }
        )

    def save_model(self, path: str) -> str:
        """Save model state to pickle file."""
        # Create models directory structure
        import foreseer_models
        base_dir = os.path.dirname(foreseer_models.__file__)
        models_dir = os.path.join(base_dir, "models", "greedy_second_best")
        os.makedirs(models_dir, exist_ok=True)

        # Add .pkl extension if not present
        if not path.endswith('.pkl'):
            path = f"{path}.pkl"

        full_path = os.path.join(models_dir, path)

        # Save state
        state = {
            'lookback': self.lookback,
            'max_long': self.max_long,
            'max_short': self.max_short,
            'umax': self.umax,
            'step_up_to': self.step_up_to,
            'net_strict': self.net_strict,
            'find_second_best': self.find_second_best,
            'min_difference': self.min_difference,
            'scale_factor': self.scale_factor,
            'last_weights': self.last_weights,
            'last_weights_second': self.last_weights_second,
            'allocations_history': self.allocations_history,
            'second_best_history': self.second_best_history
        }

        with open(full_path, 'wb') as f:
            pickle.dump(state, f)

        return full_path

    def load_model(self, path: str) -> None:
        """Load model state from pickle file."""
        # Handle path resolution
        if not os.path.isabs(path):
            import foreseer_models
            base_dir = os.path.dirname(foreseer_models.__file__)
            models_dir = os.path.join(base_dir, "models", "greedy_second_best")

            if not path.endswith('.pkl'):
                path = f"{path}.pkl"

            full_path = os.path.join(models_dir, path)
        else:
            full_path = path

        with open(full_path, 'rb') as f:
            state = pickle.load(f)

        # Restore state
        self.lookback = state['lookback']
        self.max_long = state['max_long']
        self.max_short = state['max_short']
        self.umax = state['umax']
        self.step_up_to = state['step_up_to']
        self.net_strict = state['net_strict']
        self.find_second_best = state['find_second_best']
        self.min_difference = state['min_difference']
        self.scale_factor = state['scale_factor']
        self.last_weights = state['last_weights']
        self.last_weights_second = state.get('last_weights_second')
        self.allocations_history = state.get('allocations_history', {})
        self.second_best_history = state.get('second_best_history', {})