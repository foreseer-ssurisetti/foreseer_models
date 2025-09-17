"""
Greedy integer allocation algorithm implementation.
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Any
from ..base import BaseAlgorithm, StrategyResults


class GreedyAlgorithm(BaseAlgorithm):
    """Greedy integer allocation algorithm implementation."""

    def __init__(self, name: str = "GreedyAlgorithm",
                 gross_cap: int = 20,
                 umax: int = 3,
                 step_up_to: int = 3,
                 net_strict: bool = False,
                 block: int = 30,
                 lookback: int = 24):
        super().__init__(name)
        self.gross_cap = gross_cap
        self.umax = umax
        self.step_up_to = step_up_to
        self.net_strict = net_strict
        self.block = block
        self.lookback = lookback
        self.fitted_params = {}
        self.last_allocation = None

    def _negative_sharpe(self, Xm: pd.DataFrame, u: np.ndarray) -> float:
        """Return the *negative* monthly Sharpe for a portfolio with integer units u."""
        pnl = Xm.values @ u
        mu = pnl.mean()
        sd = pnl.std(ddof=1)
        sharpe = 0.0 if sd == 0 else mu / sd
        return -sharpe

    def _greedy_integer_alloc(self, Xm: pd.DataFrame) -> np.ndarray:
        """Greedy search on integer units with configured parameters."""
        N = Xm.shape[1]
        u = np.zeros(N, dtype=int)
        base = self._negative_sharpe(Xm, u)

        iteration = 0
        max_iterations = 1000

        while iteration < max_iterations:
            iteration += 1
            best = (0.0, None, 0)
            gross_used = np.abs(u).sum()

            for i in range(N):
                head_name = self.umax - abs(u[i])
                if head_name <= 0:
                    continue

                max_step_here = min(self.step_up_to, head_name, self.gross_cap - gross_used)
                if max_step_here <= 0:
                    continue

                for sgn in (+1, -1):
                    for step in range(1, max_step_here + 1):
                        v = u.copy()
                        v[i] += sgn * step

                        if np.abs(v).sum() > self.gross_cap:
                            break

                        net_sum = v.sum()
                        net_ok = (net_sum > 0) if self.net_strict else (net_sum >= 0)
                        if not net_ok:
                            continue

                        val = self._negative_sharpe(Xm, v)
                        reduction = base - val
                        if reduction > best[0] + 1e-12:
                            best = (reduction, i, sgn * step)

            if best[1] is None:
                break

            u[best[1]] += best[2]
            base -= best[0]

            if np.abs(u).sum() >= self.gross_cap:
                break

        # Local polish
        improved = True
        polish_iterations = 0
        max_polish = 100

        while improved and polish_iterations < max_polish:
            polish_iterations += 1
            improved = False
            base = self._negative_sharpe(Xm, u)

            for i in range(N):
                for sgn in (+1, -1):
                    if abs(u[i] + sgn) > self.umax:
                        continue

                    v = u.copy()
                    v[i] += sgn

                    if np.abs(v).sum() > self.gross_cap:
                        continue

                    net_sum = v.sum()
                    net_ok = (net_sum > 0) if self.net_strict else (net_sum >= 0)
                    if not net_ok:
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

    def _to_monthly_blocks(self, returns_d: pd.DataFrame, drop_last_partial: bool = True) -> pd.DataFrame:
        """Convert daily returns into monthly blocks using block-based approach."""
        T = len(returns_d)
        month_id = (np.arange(T) // self.block) + 1
        m_ret = (1 + returns_d.set_index(month_id)).groupby(level=0).prod() - 1
        if drop_last_partial and (T % self.block) != 0:
            m_ret = m_ret.iloc[:-1]
        m_ret.index.name = "month"
        return m_ret

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
            'net_strict': self.net_strict
        }

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Produce portfolio weights for the given training window."""
        if not hasattr(self, 'fitted_params') or not self.fitted_params:
            raise ValueError("Algorithm must be fitted before prediction")

        units = self._greedy_integer_alloc(data)
        self.last_allocation = units

        total_gross = np.abs(units).sum()
        if total_gross == 0:
            weights = np.zeros(len(units))
        else:
            weights = units / total_gross

        return weights

    def backtest(self, returns_d: pd.DataFrame) -> StrategyResults:
        """Perform walk-forward backtest using the exact greedy algorithm logic."""
        Xm = self._to_monthly_blocks(returns_d, drop_last_partial=True)
        months = Xm.index

        if len(months) <= self.lookback + 1:
            raise ValueError(f"Insufficient data: need at least {self.lookback + 2} months, got {len(months)}")

        results = []
        allocs = {}

        for k in range(self.lookback, len(months)):
            if k == len(months) - 1:
                break

            train = Xm.iloc[k-self.lookback:k]
            hold = Xm.iloc[k]

            # Use the exact greedy integer allocation
            u = self._greedy_integer_alloc(train)
            allocs[months[k]] = u

            pnl_hold = float(hold.values @ u)

            # For reporting, convert back to positive Sharpe
            train_sharpe = -self._negative_sharpe(train, u)

            results.append({
                "month": months[k],
                "pnl": pnl_hold,
                "sh_train": train_sharpe,  # Positive Sharpe for reporting
                "gross_units": int(np.abs(u).sum()),
                "net_units": int(u.sum())
            })

        res = pd.DataFrame(results).set_index("month")

        if len(res) == 0:
            return StrategyResults(
                strategy_name=self.name,
                weights=np.array([]),
                oos_sharpe_daily=0.0,
                oos_sharpe_annual=0.0,
                allocations_best=pd.DataFrame(),
                oos_returns=pd.Series(dtype=float),
                training_metrics={
                    "monthly_returns": Xm,
                    "allocations_by_month": allocs,
                    "oos_sharpe_monthly": 0.0,
                    "oos_sharpe_annual": 0.0,
                    "no_data": True
                }
            )

        mu_oos = res["pnl"].mean()
        sd_oos = res["pnl"].std(ddof=1)
        sharpe_oos = 0.0 if sd_oos == 0 else mu_oos / sd_oos

        # Get final allocation weights
        if len(allocs) > 0:
            # Use the last allocation
            last_month = list(allocs.keys())[-1]
            last_allocation = allocs[last_month]
            total_gross = np.abs(last_allocation).sum()
            final_weights = last_allocation / total_gross if total_gross > 0 else last_allocation
        else:
            final_weights = np.array([])

        # Create allocations DataFrame using the actual asset names
        if len(Xm.columns) == len(final_weights):
            allocations_df = pd.DataFrame({
                'strat': list(Xm.columns),
                'weight': final_weights
            })
        else:
            allocations_df = pd.DataFrame()

        return StrategyResults(
            strategy_name=self.name,
            weights=final_weights,
            oos_sharpe_daily=sharpe_oos / np.sqrt(30),  # Rough daily conversion
            oos_sharpe_annual=sharpe_oos * np.sqrt(12),
            allocations_best=allocations_df,
            oos_returns=res["pnl"],
            training_metrics={
                "monthly_returns": Xm,
                "allocations_by_month": allocs,
                "oos": res,
                "oos_sharpe_monthly": sharpe_oos,
                "total_periods": len(res),
                "avg_gross_units": res["gross_units"].mean(),
                "avg_net_units": res["net_units"].mean(),
                "avg_train_sharpe": res["sh_train"].mean()
            }
        )

    def save_model(self, path: str) -> str:
        """Save the greedy algorithm parameters and fitted state."""
        if not path.endswith('.pkl'):
            path = f"{path}.pkl"

        model_state = {
            'name': self.name,
            'gross_cap': self.gross_cap,
            'umax': self.umax,
            'step_up_to': self.step_up_to,
            'net_strict': self.net_strict,
            'block': self.block,
            'lookback': self.lookback,
            'fitted_params': self.fitted_params,
            'last_allocation': self.last_allocation
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(model_state, f)

        return path

    def load_model(self, path: str) -> None:
        """Load the greedy algorithm parameters and fitted state."""
        with open(path, 'rb') as f:
            model_state = pickle.load(f)

        self.name = model_state['name']
        self.gross_cap = model_state['gross_cap']
        self.umax = model_state['umax']
        self.step_up_to = model_state['step_up_to']
        self.net_strict = model_state['net_strict']
        self.block = model_state['block']
        self.lookback = model_state['lookback']
        self.fitted_params = model_state['fitted_params']
        self.last_allocation = model_state['last_allocation']
