"""Hyperparameter sweep — launch multiple runs, then compare in the dashboard.

Trains the same linear regression model with different learning rates,
producing multiple runs you can compare side-by-side.

Usage:
    python examples/hyperparameter_sweep.py
    waddle ls                    # see all 4 runs
    waddle serve                 # select multiple runs → "Compare Selected"
"""

import random
import waddle


def generate_data(n: int, noise: float) -> list:
    data = []
    for _ in range(n):
        x = random.uniform(-4, 4)
        y = 2.0 * x + 1.0 + random.gauss(0, noise)
        data.append((x, y))
    return data


def train_run(data, lr, epochs, run_name):
    """Train one model and log to waddle."""
    with waddle.init(
        project="hp-sweep",
        name=run_name,
        config={"lr": lr, "epochs": epochs, "samples": len(data)},
        tags={"sweep": "learning_rate", "model": "linear"},
        system_metrics=False,
    ):
        w = b = 0.0
        n = len(data)

        for epoch in range(epochs):
            gw = sum((w * x + b - y) * x for x, y in data) * (2.0 / n)
            gb = sum((w * x + b - y) for x, y in data) * (2.0 / n)
            w -= lr * gw
            b -= lr * gb
            mse = sum((w * x + b - y) ** 2 for x, y in data) / n
            waddle.log({"mse": mse, "weight": w, "bias": b})

        final_mse = sum((w * x + b - y) ** 2 for x, y in data) / n
        waddle.log_param("final_mse", round(final_mse, 4))
        return w, b, final_mse


def main():
    random.seed(42)
    data = generate_data(200, noise=0.3)

    learning_rates = [0.001, 0.01, 0.05, 0.1]
    epochs = 80

    print(f"Running sweep: {len(learning_rates)} configs")
    print("-" * 50)

    results = []
    for lr in learning_rates:
        name = f"lr={lr}"
        w, b, mse = train_run(data, lr, epochs, name)
        results.append((lr, w, b, mse))
        print(f"  {name:10s} -> w={w:.4f}  b={b:.4f}  mse={mse:.4f}")

    print("-" * 50)
    best = min(results, key=lambda r: r[3])
    print(f"Best: lr={best[0]} with MSE={best[3]:.4f}")
    print(f"\nRun 'waddle serve' to compare all runs in the dashboard.")


if __name__ == "__main__":
    main()
