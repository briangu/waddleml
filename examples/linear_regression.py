"""Linear regression with full WaddleML instrumentation.

Trains y = 2x + 1 with batch gradient descent and logs everything:
config, per-epoch metrics, final evaluation, tags, and a model artifact.

Usage:
    python examples/linear_regression.py
    python examples/linear_regression.py --epochs 100 --lr 0.03
    waddle serve
"""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import waddle


def generate_data(n: int, noise: float) -> list:
    """Synthetic data: y = 2x + 1 + noise."""
    data = []
    for _ in range(n):
        x = random.uniform(-4, 4)
        y = 2.0 * x + 1.0 + random.gauss(0, noise)
        data.append((x, y))
    return data


def train(data, epochs, lr):
    """Batch gradient descent for y = wx + b."""
    w = random.uniform(-0.5, 0.5)
    b = random.uniform(-0.5, 0.5)
    n = len(data)

    for epoch in range(epochs):
        gw = sum((w * x + b - y) * x for x, y in data) * (2.0 / n)
        gb = sum((w * x + b - y) for x, y in data) * (2.0 / n)
        w -= lr * gw
        b -= lr * gb

        mse = sum((w * x + b - y) ** 2 for x, y in data) / n
        mae = sum(abs(w * x + b - y) for x, y in data) / n
        waddle.log({"train/mse": mse, "train/mae": mae, "train/weight": w, "train/bias": b})

    return w, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with waddle.init(
        project="linear-regression",
        name=f"lr={args.lr}_epochs={args.epochs}",
        config={"lr": args.lr, "epochs": args.epochs, "samples": args.samples, "noise": args.noise, "seed": args.seed},
        tags={"model": "linear", "task": "regression"},
    ):
        data = generate_data(args.samples, args.noise)
        w, b = train(data, args.epochs, args.lr)

        final_mse = sum((w * x + b - y) ** 2 for x, y in data) / len(data)
        waddle.log({"eval/final_mse": final_mse})

        # save model artifact
        model = {"weight": w, "bias": b}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(model, f, indent=2)
            tmp_path = f.name
        waddle.log_artifact("model.json", tmp_path, kind="model", inline=True)
        Path(tmp_path).unlink()

        print(f"Learned: y = {w:.4f}x + {b:.4f}  (true: y = 2x + 1)")
        print(f"Final MSE: {final_mse:.4f}")


if __name__ == "__main__":
    main()
