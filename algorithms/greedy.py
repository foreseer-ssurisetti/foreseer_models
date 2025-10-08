"""
Greedy algorithm
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Optional, Dict, List, Tuple
from ..base import (BaseAlgorithm, StrategyResults, AllocationSnapshot, PositionDetail,
                    ModelSpecification, PerformanceMetrics)


class GreedyAlgorithm(BaseAlgorithm):
    """
    Greedy integer allocation optimizer
    """

    def __init__(self,
                 lookback: int = 24,
                 max_net: int = 6,
                 max_gross: int = 10,
                 n_outright: int = 90,
                 step_up_to: int = 5,
                 find_second_best: bool = True,
                 min_difference: int = 2,
                 strat_legs_file: Optional[str] = None):
        super().__init__("greedy")

        self.lookback = lookback
        self.max_net = max_net
        self.max_gross = max_gross
        self.n_outright = n_outright
        self.step_up_to = step_up_to
        self.find_second_best = find_second_best
        self.min_difference = min_difference
        self.strat_legs_file = strat_legs_file

        self.strategy_legs_info = None
        if strat_legs_file:
            self.strategy_legs_info = self._load_strategy_legs(strat_legs_file)

    def _load_strategy_legs(self, filepath: str) -> Dict[int, Dict]:
        """Load strategy leg information from Excel file."""
        try:
            strat_df = pd.read_excel(filepath)
            strategy_info = {}

            for _, row in strat_df.iterrows():
                strat_id = int(row['ID'])
                legs = []

                for leg_num in range(1, 5):
                    leg_col = f'L{leg_num}'
                    size_col = f'L{leg_num}_Siz'

                    if leg_col in row and row[leg_col] and str(row[leg_col]).strip():
                        leg_desc = str(row[leg_col]).strip()
                        leg_size = float(row[size_col]) if size_col in row else 0

                        if leg_size != 0:
                            legs.append({
                                'description': leg_desc,
                                'size': leg_size,
                                'direction': 'LONG' if leg_size > 0 else 'SHORT'
                            })

                strategy_info[strat_id] = {
                    'num_legs': len(legs),
                    'legs': legs
                }

            return strategy_info
        except Exception as e:
            print(f"Warning: Could not load strategy legs from {filepath}: {e}")
            return None

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

    def _check_constraints(self, u: np.ndarray) -> bool:
        """Check if allocation satisfies constraints."""
        if np.any(u[:self.n_outright] < -1):
            return False

        if len(u) > self.n_outright:
            if np.any(u[self.n_outright:] < 0):
                return False

        net = u.sum()
        if net < 0 or net > self.max_net:
            return False

        gross = np.abs(u).sum()
        if gross > self.max_gross:
            return False

        return True

    def fit(self, Xm: pd.DataFrame,
            exclude_allocation: Optional[np.ndarray] = None) -> np.ndarray:
        """Greedy search for optimal integer allocation."""
        N = Xm.shape[1]
        u = np.zeros(N, dtype=int)

        base = self._negative_sharpe(Xm, u)
        iteration = 0
        max_iterations = 2500

        while iteration < max_iterations:
            iteration += 1
            best = (0.0, None, 0)

            for i in range(N):
                min_allowed = -1 if i < self.n_outright else 0

                for sgn in (+1, -1):
                    if sgn < 0 and u[i] <= min_allowed:
                        continue

                    for step in range(1, self.step_up_to + 1):
                        v = u.copy()
                        v[i] += sgn * step

                        if not self._check_constraints(v):
                            break

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

            if u.sum() >= self.max_net or np.abs(u).sum() >= self.max_gross:
                break

        # Local polish
        improved = True
        polish_iter = 0

        while improved and polish_iter < 100:
            polish_iter += 1
            improved = False
            base = self._negative_sharpe(Xm, u)

            for i in range(N):
                for sgn in (+1, -1):
                    v = u.copy()
                    v[i] += sgn

                    if not self._check_constraints(v):
                        continue

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

    def _find_top_k_allocations(self, Xm: pd.DataFrame, k: int = 2) -> List[Tuple[float, np.ndarray]]:
        """
        Find top k different allocations, sorted by performance (best first).
        Returns list of (negative_sharpe, allocation) tuples.
        """
        solutions = []

        for rank in range(k):
            if rank == 0:
                # Find first allocation
                u = self.fit(Xm, exclude_allocation=None)
            else:
                # Find next allocation different from the first one
                u = self.fit(Xm, exclude_allocation=solutions[0][1])

            score = self._negative_sharpe(Xm, u)
            solutions.append((score, u.copy()))

        # Sort by negative_sharpe (ascending = best first)
        # Smallest negative_sharpe = largest positive sharpe
        solutions.sort(key=lambda x: x[0])

        return solutions

    def _calculate_leg_stats(self, allocation: np.ndarray) -> Tuple[int, int, int]:
        """Calculate total, long, and short legs for an allocation."""
        if not self.strategy_legs_info:
            return None, None, None

        total_legs = 0
        long_legs = 0
        short_legs = 0

        for i, units in enumerate(allocation):
            if units != 0:
                strat_id = i + 1
                if strat_id in self.strategy_legs_info:
                    num_legs = self.strategy_legs_info[strat_id]['num_legs']
                    total_legs += abs(units) * num_legs
                    if units > 0:
                        long_legs += units * num_legs
                    else:
                        short_legs += abs(units) * num_legs

        return total_legs, long_legs, short_legs

    def _create_allocation_snapshot(self, allocation: np.ndarray,
                                   date: pd.Timestamp,
                                   train_sharpe: Optional[float] = None,
                                   training_start: Optional[pd.Timestamp] = None,
                                   training_end: Optional[pd.Timestamp] = None) -> AllocationSnapshot:
        """Create an AllocationSnapshot from an allocation vector."""
        positions = []

        for i, units in enumerate(allocation):
            if units != 0:
                strat_id = i + 1
                pos_info = {
                    'strategy_id': strat_id,
                    'strategy_name': f'Strategy_{strat_id}',
                    'units': int(units),
                    'direction': 'LONG' if units > 0 else 'SHORT',
                }

                if self.strategy_legs_info and strat_id in self.strategy_legs_info:
                    strat_info = self.strategy_legs_info[strat_id]
                    pos_info['num_legs'] = strat_info['num_legs']
                    pos_info['leg_details'] = strat_info['legs']

                positions.append(PositionDetail(**pos_info))

        long_units = int(allocation[allocation > 0].sum())
        short_units = int(np.abs(allocation[allocation < 0]).sum())
        total_legs, long_legs, short_legs = self._calculate_leg_stats(allocation)

        return AllocationSnapshot(
            date=date,
            allocation_vector=allocation.copy(),
            positions=positions,
            long_units=long_units,
            short_units=short_units,
            gross_units=long_units + short_units,
            net_units=int(allocation.sum()),
            total_legs=total_legs,
            long_legs=long_legs,
            short_legs=short_legs,
            train_sharpe=train_sharpe,
            training_start=training_start,
            training_end=training_end,
        )

    def _calculate_performance_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """Calculate performance metrics from returns series."""
        mu = returns.mean()
        sd = returns.std(ddof=1)
        sharpe_monthly = 0.0 if sd == 0 else mu / sd
        sharpe_annual = sharpe_monthly * np.sqrt(12)

        wins = (returns > 0).sum()
        total = len(returns)
        win_rate = wins / total if total > 0 else 0.0

        avg_win = returns[returns > 0].mean() if wins > 0 else 0.0
        avg_loss = returns[returns < 0].mean() if (total - wins) > 0 else 0.0

        return PerformanceMetrics(
            oos_sharpe_monthly=sharpe_monthly,
            oos_sharpe_annual=sharpe_annual,
            mean_return=mu,
            std_return=sd,
            median_return=returns.median(),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )


    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate allocation for the given data."""
        Xm = self._to_monthly_blocks(data, drop_last_partial=True)

        if len(Xm) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} months, got {len(Xm)}")

        train = Xm.iloc[-self.lookback:]
        allocation = self.fit(train)

        return allocation

    def backtest(self, data: pd.DataFrame) -> StrategyResults:
        """Perform walk-forward backtest."""
        Xm = self._to_monthly_blocks(data, drop_last_partial=True)
        dates = Xm.index

        if len(dates) <= self.lookback + 1:
            raise ValueError(
                f"Insufficient data: need at least {self.lookback + 2} months, "
                f"got {len(dates)} months"
            )

        allocation_snapshots_best = {}
        allocation_snapshots_second = {}
        allocations_best_list = []
        allocations_second_list = []
        pnls_best_list = []
        pnls_second_list = []
        train_sharpes_best = []
        train_sharpes_second = []
        results_records = []

        # Walk-forward backtest
        for k in range(self.lookback, len(dates) - 1):
            train = Xm.iloc[k - self.lookback:k]
            hold = Xm.iloc[k]
            hold_date = dates[k]

            if self.find_second_best:
                # Find and sort top 2 allocations
                solutions = self._find_top_k_allocations(train, k=2)
                u_best = solutions[0][1]
                u_second = solutions[1][1]
                train_sharpe_best = -solutions[0][0]
                train_sharpe_second = -solutions[1][0]
            else:
                u_best = self.fit(train)
                train_sharpe_best = -self._negative_sharpe(train, u_best)
                u_second = None
                train_sharpe_second = None

            pnl_best = float(hold.values @ u_best)
            pnl_second = float(hold.values @ u_second) if u_second is not None else None

            snapshot_best = self._create_allocation_snapshot(
                u_best, hold_date, train_sharpe_best,
                train.index[0], train.index[-1]
            )
            allocation_snapshots_best[hold_date] = snapshot_best
            allocations_best_list.append(u_best)
            pnls_best_list.append(pnl_best)
            train_sharpes_best.append(train_sharpe_best)

            if self.find_second_best:
                snapshot_second = self._create_allocation_snapshot(
                    u_second, hold_date, train_sharpe_second,
                    train.index[0], train.index[-1]
                )
                allocation_snapshots_second[hold_date] = snapshot_second
                allocations_second_list.append(u_second)
                pnls_second_list.append(pnl_second)
                train_sharpes_second.append(train_sharpe_second)

            record = {
                'date': hold_date,
                'pnl': pnl_best,
                'train_sharpe': train_sharpe_best,
                'long_units': snapshot_best.long_units,
                'short_units': snapshot_best.short_units,
                'net_units': snapshot_best.net_units,
                'gross_units': snapshot_best.gross_units,
                'num_positions': len(snapshot_best.positions),
            }

            if snapshot_best.total_legs is not None:
                record['total_legs'] = snapshot_best.total_legs
                record['long_legs'] = snapshot_best.long_legs
                record['short_legs'] = snapshot_best.short_legs

            if self.find_second_best:
                record['pnl_second'] = pnl_second
                record['train_sharpe_second'] = train_sharpe_second
                record['sharpe_diff'] = train_sharpe_best - train_sharpe_second

            results_records.append(record)

        results_table = pd.DataFrame(results_records).set_index('date')
        backtest_dates = [rec['date'] for rec in results_records]
        allocations_best_df = pd.DataFrame(allocations_best_list, index=backtest_dates)
        oos_returns_best = pd.Series(pnls_best_list, index=backtest_dates)
        performance_best = self._calculate_performance_metrics(oos_returns_best)

        allocations_second_df = None
        oos_returns_second = None
        performance_second = None
        sharpe_diff_history = None

        if self.find_second_best and allocations_second_list:
            allocations_second_df = pd.DataFrame(allocations_second_list, index=backtest_dates)
            oos_returns_second = pd.Series(pnls_second_list, index=backtest_dates)
            performance_second = self._calculate_performance_metrics(oos_returns_second)
            sharpe_diff_history = pd.Series(
                [b - s for b, s in zip(train_sharpes_best, train_sharpes_second)],
                index=backtest_dates
            )

        # Generate next month allocation
        train_final = Xm.iloc[-self.lookback:]
        next_month = dates[-1] + pd.DateOffset(months=1)
        next_month = next_month + pd.offsets.MonthEnd(0)

        if self.find_second_best:
            solutions = self._find_top_k_allocations(train_final, k=2)
            u_next_best = solutions[0][1]
            u_next_second = solutions[1][1]
            train_sharpe_next_best = -solutions[0][0]
            train_sharpe_next_second = -solutions[1][0]
        else:
            u_next_best = self.fit(train_final)
            train_sharpe_next_best = -self._negative_sharpe(train_final, u_next_best)
            u_next_second = None
            train_sharpe_next_second = None

        next_month_snapshot_best = self._create_allocation_snapshot(
            u_next_best, next_month, train_sharpe_next_best,
            train_final.index[0], train_final.index[-1]
        )

        next_month_snapshot_second = None
        if self.find_second_best:
            next_month_snapshot_second = self._create_allocation_snapshot(
                u_next_second, next_month, train_sharpe_next_second,
                train_final.index[0], train_final.index[-1]
            )

        model_spec = ModelSpecification(
            algorithm_name=self.name,
            lookback=self.lookback,
            max_net=self.max_net,
            max_gross=self.max_gross,
            n_outright=self.n_outright,
            training_start=train_final.index[0],
            training_end=train_final.index[-1],
            backtest_start=backtest_dates[0],
            backtest_end=backtest_dates[-1],
            find_second_best=self.find_second_best,
            step_up_to=self.step_up_to,
            min_difference=self.min_difference,
        )

        results = StrategyResults(
            model_spec=model_spec,
            strategy_legs_info=self.strategy_legs_info,
            performance_metrics=performance_best,
            performance_metrics_second=performance_second,
            oos_returns=oos_returns_best,
            oos_returns_second=oos_returns_second,
            allocations_best=allocations_best_df,
            allocations_second_best=allocations_second_df,
            allocation_snapshots=allocation_snapshots_best,
            allocation_snapshots_second=allocation_snapshots_second if self.find_second_best else None,
            results_table=results_table,
            next_month_allocation=next_month_snapshot_best,
            next_month_second_best=next_month_snapshot_second,
            weights=u_next_best,
            weights_second_best=u_next_second,
            sharpe_diff_history=sharpe_diff_history,
        )

        return results

    def save_model(self, path: str) -> str:
        """Save model parameters and configuration."""
        try:
            import foreseer_models
            foreseer_dir = os.path.dirname(foreseer_models.__file__)
        except ImportError:
            foreseer_dir = os.path.dirname(os.path.dirname(__file__))

        algo_dir = os.path.join(foreseer_dir, "models", self.name)
        os.makedirs(algo_dir, exist_ok=True)

        if not path.endswith('.pkl'):
            path = f"{path}.pkl"

        full_path = os.path.join(algo_dir, path)

        state = {
            'name': self.name,
            'lookback': self.lookback,
            'max_net': self.max_net,
            'max_gross': self.max_gross,
            'n_outright': self.n_outright,
            'step_up_to': self.step_up_to,
            'find_second_best': self.find_second_best,
            'min_difference': self.min_difference,
            'strat_legs_file': self.strat_legs_file,
            'strategy_legs_info': self.strategy_legs_info
        }

        with open(full_path, 'wb') as f:
            pickle.dump(state, f)

        return full_path

    def load_model(self, path: str) -> None:
        """Load model parameters and configuration."""
        if not os.path.isabs(path):
            try:
                import foreseer_models
                foreseer_dir = os.path.dirname(foreseer_models.__file__)
            except ImportError:
                foreseer_dir = os.path.dirname(os.path.dirname(__file__))

            algo_dir = os.path.join(foreseer_dir, "models", self.name)

            if not path.endswith('.pkl'):
                path = f"{path}.pkl"

            path = os.path.join(algo_dir, path)

        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.name = state['name']
        self.lookback = state['lookback']
        self.max_net = state['max_net']
        self.max_gross = state['max_gross']
        self.n_outright = state['n_outright']
        self.step_up_to = state['step_up_to']
        self.find_second_best = state['find_second_best']
        self.min_difference = state['min_difference']
        self.strat_legs_file = state.get('strat_legs_file')
        self.strategy_legs_info = state.get('strategy_legs_info')