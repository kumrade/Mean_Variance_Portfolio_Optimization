"""Constraint-aware minimum-variance and mean-variance research engine."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

TRADING_DAYS = 252


@dataclass(frozen=True)
class Rules:
    objective: str
    min_weight: float
    max_weight: float
    allow_short: bool
    gross_limit: float = 1.0
    target_quantile: float = 0.65

    @property
    def name(self) -> str:
        side = "Long Short" if self.allow_short else "Long Only"
        return (f"{self.objective} | {side} | min {self.min_weight:.0%} | "
                f"max {self.max_weight:.0%} | gross {self.gross_limit:.0%}")

    def bounds(self, n: int) -> list[tuple[float, float]]:
        return [(-self.max_weight, self.max_weight)] * n if self.allow_short else [(self.min_weight, self.max_weight)] * n

    def feasible(self, n: int) -> bool:
        if self.max_weight <= 0 or self.max_weight * n < 1 - 1e-9:
            return False
        if self.allow_short:
            return self.gross_limit >= 1 and self.min_weight == 0
        return 0 <= self.min_weight <= self.max_weight and self.min_weight * n <= 1 + 1e-9


@dataclass
class Fit:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


def demo_prices(symbols: list[str], sessions: int = 650, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    annual_return = np.linspace(0.09, 0.16, len(symbols))
    annual_vol = np.linspace(0.18, 0.30, len(symbols))
    beta = rng.uniform(0.35, 0.85, len(symbols))
    corr = np.outer(beta, beta); np.fill_diagonal(corr, 1)
    cov = corr * np.outer(annual_vol, annual_vol) / TRADING_DAYS
    shocks = rng.standard_normal((sessions, len(symbols))) @ np.linalg.cholesky(cov + np.eye(len(symbols)) * 1e-12).T
    log_returns = (annual_return - 0.5 * annual_vol**2) / TRADING_DAYS + shocks
    return pd.DataFrame(100 * np.exp(np.cumsum(log_returns, axis=0)),
                        index=pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=sessions), columns=symbols)


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    clean = prices.sort_index().apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return clean.pct_change(fill_method=None).dropna()


def estimates(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    cov = (cov + cov.T) / 2
    values, vectors = np.linalg.eigh(cov.values)
    cov.loc[:, :] = vectors @ np.diag(np.maximum(values, 1e-8)) @ vectors.T
    return mu, cov


def performance(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    ret = float(w @ mu); vol = float(np.sqrt(max(w @ cov @ w, 0)))
    return ret, vol, (ret - rf) / vol if vol > 1e-12 else -np.inf


def feasible_return_range(mu: pd.Series, rules: Rules) -> tuple[float, float]:
    n = len(mu)
    if not rules.feasible(n): raise ValueError("Infeasible position constraints")
    if rules.allow_short:
        c = np.r_[mu.values, -mu.values]
        kwargs = dict(A_eq=np.r_[np.ones(n), -np.ones(n)][None, :], b_eq=[1],
                      A_ub=np.ones((1, 2*n)), b_ub=[rules.gross_limit],
                      bounds=[(0, rules.max_weight)] * (2*n), method="highs")
        lo, hi = linprog(c, **kwargs), linprog(-c, **kwargs)
    else:
        kwargs = dict(A_eq=np.ones((1, n)), b_eq=[1], bounds=rules.bounds(n), method="highs")
        lo, hi = linprog(mu.values, **kwargs), linprog(-mu.values, **kwargs)
    if not lo.success or not hi.success: raise ValueError("No feasible return range")
    return float(lo.fun), float(-hi.fun)


def solve(mu: pd.Series, cov: pd.DataFrame, rules: Rules, rf: float = 0.065, target: float | None = None) -> Fit:
    n = len(mu)
    if not rules.feasible(n): raise ValueError("Infeasible rules")
    m, s = mu.values, cov.values
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if rules.allow_short:
        constraints.append({"type": "ineq", "fun": lambda w: rules.gross_limit - np.sum(np.abs(w))})
    if target is not None:
        lo, hi = feasible_return_range(mu, rules)
        if not lo - 1e-7 <= target <= hi + 1e-7: raise ValueError("Target outside feasible range")
        constraints.append({"type": "eq", "fun": lambda w, t=target: w @ m - t})
    variance = lambda w: float(w @ s @ w)
    def neg_sharpe(w): return -performance(w, m, s, rf)[2]
    objective = variance if rules.objective in ("Minimum Variance", "Target Return") else neg_sharpe
    rng = np.random.default_rng(123)
    starts = [np.repeat(1/n, n)] + [rng.dirichlet(np.ones(n)) for _ in range(12)]
    best = None
    for start in starts:
        if rules.allow_short: start = np.repeat(1/n, n)
        result = minimize(objective, start, method="SLSQP", bounds=rules.bounds(n), constraints=constraints,
                          options={"maxiter": 800, "ftol": 1e-11})
        if result.success and np.all(np.isfinite(result.x)) and (best is None or result.fun < best.fun): best = result
    if best is None: raise ValueError("Optimizer could not find a feasible solution")
    ret, vol, sharpe = performance(best.x, m, s, rf)
    return Fit(pd.Series(best.x, index=mu.index), ret, vol, sharpe)


def fit_rule(mu: pd.Series, cov: pd.DataFrame, rules: Rules, rf: float) -> Fit:
    target = None
    if rules.objective == "Target Return":
        lo, hi = feasible_return_range(mu, rules); target = lo + rules.target_quantile * (hi - lo)
    return solve(mu, cov, rules, rf, target)


def drawdown(series: pd.Series) -> pd.Series:
    wealth = (1 + series).cumprod(); return wealth / wealth.cummax() - 1


def walk_forward(daily: pd.DataFrame, rules: Rules, rf: float = 0.065,
                 train: int = 252, test: int = 63, step: int = 63) -> dict:
    chunks, turnovers, previous = [], [], None
    for end in range(train, len(daily) - test + 1, step):
        mu, cov = estimates(daily.iloc[end-train:end]); fit = fit_rule(mu, cov, rules, rf)
        if previous is not None: turnovers.append(float(np.abs(fit.weights.values - previous).sum()))
        previous = fit.weights.values; chunks.append(daily.iloc[end:end+test] @ fit.weights)
    if not chunks: raise ValueError("Not enough observations for walk-forward evaluation")
    series = pd.concat(chunks).sort_index(); n = len(series)
    ann_ret = float((1 + series).prod() ** (TRADING_DAYS/n) - 1)
    ann_vol = float(series.std(ddof=1) * np.sqrt(TRADING_DAYS)); downside = float(series[series < 0].std(ddof=1) * np.sqrt(TRADING_DAYS))
    tail = float(series.quantile(0.05)); cvar = float(-series[series <= tail].mean() * np.sqrt(TRADING_DAYS))
    return {"oos_return": ann_ret, "oos_vol": ann_vol, "sharpe": (ann_ret-rf)/ann_vol,
            "sortino": (ann_ret-rf)/downside, "max_drawdown": float(drawdown(series).min()),
            "cvar": cvar, "turnover": float(np.mean(turnovers)) if turnovers else 0, "observations": n}


def candidate_rules(n_assets: int) -> list[Rules]:
    candidates = []
    for objective, max_w, min_w in product(["Maximum Sharpe", "Minimum Variance", "Target Return"],
                                            sorted({max(0.25, 1/n_assets), 0.40, 0.60}), [0, 0.02]):
        rule = Rules(objective, min_w, max_w, False)
        if rule.feasible(n_assets): candidates.append(rule)
    for objective, max_w, gross in product(["Maximum Sharpe", "Minimum Variance", "Target Return"],
                                            sorted({max(0.25, 1/n_assets), 0.40, 0.60}), [1.25, 1.50]):
        rule = Rules(objective, 0, max_w, True, gross)
        if rule.feasible(n_assets): candidates.append(rule)
    return candidates


def rank_tournament(daily: pd.DataFrame, rf: float = 0.065, train: int = 252, test: int = 63, step: int = 63) -> pd.DataFrame:
    rows = []
    for rules in candidate_rules(daily.shape[1]):
        try: rows.append({"rules": rules, "configuration": rules.name, **walk_forward(daily, rules, rf, train, test, step)})
        except ValueError: continue
    table = pd.DataFrame(rows)
    if table.empty: raise ValueError("Every candidate failed")
    table["score"] = (0.30*table.sharpe.rank(pct=True) + 0.15*table.sortino.rank(pct=True) +
                      0.15*(-table.max_drawdown).rank(pct=True) + 0.10*(-table.cvar).rank(pct=True) +
                      0.10*(-table.oos_vol).rank(pct=True) + 0.10*(-table.turnover).rank(pct=True) +
                      0.10*table.oos_return.rank(pct=True))
    table = table.sort_values(["score", "sharpe"], ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table)+1))
    table["investable"] = (table.sharpe > 0) & (table.oos_return > rf) & (table.max_drawdown > -0.25)
    return table


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, rules: Rules, rf: float = 0.065, points: int = 45) -> pd.DataFrame:
    minimum = solve(mu, cov, replace(rules, objective="Minimum Variance"), rf)
    _, maximum = feasible_return_range(mu, rules); rows = []
    target_rules = replace(rules, objective="Target Return")
    for target in np.linspace(minimum.expected_return, maximum, points):
        try:
            fit = solve(mu, cov, target_rules, rf, float(target))
            rows.append({"return": fit.expected_return, "volatility": fit.volatility, "sharpe": fit.sharpe})
        except ValueError: pass
    return pd.DataFrame(rows)
