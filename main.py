"""Run the Minimum Variance Portfolio research workflow."""
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from src.minimum_variance import demo_prices, efficient_frontier, estimates, fit_rule, rank_tournament, returns_from_prices


def run(price_csv: str | None, output: Path, rf: float) -> None:
    symbols = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E"]
    prices = pd.read_csv(price_csv, index_col=0, parse_dates=True) if price_csv else demo_prices(symbols)
    daily = returns_from_prices(prices)
    ranking = rank_tournament(daily, rf=rf)
    leader = ranking.iloc[0]
    mu, cov = estimates(daily)
    fit = fit_rule(mu, cov, leader.rules, rf)
    frontier = efficient_frontier(mu, cov, leader.rules, rf)
    output.mkdir(parents=True, exist_ok=True)
    ranking.drop(columns="rules").to_csv(output / "ranking.csv", index=False)
    fit.weights.rename("weight").to_csv(output / "weights.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(frontier.volatility, frontier["return"], lw=2)
    axes[0].scatter([fit.volatility], [fit.expected_return], marker="*", s=180, color="#e0a800", edgecolor="black")
    axes[0].set(title="Constraint Consistent Efficient Frontier", xlabel="Annualized volatility", ylabel="Annualized expected return")
    positive = fit.weights[fit.weights > 1e-6]
    axes[1].pie(positive, labels=positive.index, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Selected Portfolio Allocation")
    fig.tight_layout(); fig.savefig(output / "frontier_and_allocation.png", dpi=180); plt.close(fig)

    status = "INVESTABLE" if bool(leader.investable) else "NO TRADE - diagnostic leader only"
    print(f"{status}\n{leader.configuration}\nOOS score: {leader.score:.3f}\n{fit.weights.round(4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimum Variance Portfolio tournament")
    parser.add_argument("--prices", help="CSV with dates in the first column and one price column per asset")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--risk-free-rate", type=float, default=0.065)
    args = parser.parse_args(); run(args.prices, Path(args.output), args.risk_free_rate)
