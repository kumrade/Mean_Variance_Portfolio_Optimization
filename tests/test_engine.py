import numpy as np
from src.minimum_variance import Rules, demo_prices, efficient_frontier, estimates, rank_tournament, returns_from_prices, solve


def test_constraints_and_frontier():
    returns = returns_from_prices(demo_prices(["A", "B", "C", "D", "E"], seed=21))
    mu, cov = estimates(returns)
    rules = Rules("Minimum Variance", 0, 0.40, False)
    fit = solve(mu, cov, rules)
    assert np.isclose(fit.weights.sum(), 1, atol=1e-5)
    assert fit.weights.min() >= -1e-7 and fit.weights.max() <= 0.40001
    frontier = efficient_frontier(mu, cov, rules)
    assert len(frontier) >= 30 and frontier.volatility.notna().all()


def test_tournament_marks_investability():
    daily = returns_from_prices(demo_prices(["A", "B", "C", "D", "E"], seed=11))
    ranking = rank_tournament(daily)
    assert not ranking.empty
    assert {"score", "investable", "max_drawdown", "cvar", "turnover"}.issubset(ranking.columns)
