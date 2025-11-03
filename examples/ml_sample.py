"""Example linear regression training loop instrumented with Waddle logging.

Run directly with `python3 -m examples.ml_sample --epochs 50` or through the
Waddle CLI:
`python3 waddle/waddle_cli.py run --repo main --entry examples.ml_sample -- --epochs 50`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a toy linear model and log to Waddle.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of gradient descent steps.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Gradient descent learning rate.")
    parser.add_argument("--samples", type=int, default=200, help="Number of synthetic training points.")
    parser.add_argument("--noise", type=float, default=0.3, help="Standard deviation of Gaussian noise.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for repeatability.")
    return parser.parse_args(list(argv))


def make_regression_data(count: int, noise: float) -> List[Tuple[float, float]]:
    """Generate y = 2x + 1 with configurable Gaussian noise."""
    points: List[Tuple[float, float]] = []
    for _ in range(count):
        x = random.uniform(-4.0, 4.0)
        y = 2.0 * x + 1.0 + random.gauss(0.0, noise)
        points.append((x, y))
    return points


def mean_squared_error(model: Tuple[float, float], data: Iterable[Tuple[float, float]]) -> float:
    w, b = model
    total = 0.0
    n = 0
    for x, y in data:
        pred = w * x + b
        total += (pred - y) ** 2
        n += 1
    return total / max(n, 1)


def mean_absolute_error(model: Tuple[float, float], data: Iterable[Tuple[float, float]]) -> float:
    w, b = model
    total = 0.0
    n = 0
    for x, y in data:
        pred = w * x + b
        total += abs(pred - y)
        n += 1
    return total / max(n, 1)


def train_linear_model(
    data: List[Tuple[float, float]],
    epochs: int,
    learning_rate: float,
    run,
) -> Tuple[float, float]:
    """Simple batch gradient descent for y = wx + b."""
    w = random.uniform(-0.5, 0.5)
    b = random.uniform(-0.5, 0.5)
    n = float(len(data)) or 1.0

    for epoch in range(epochs):
        grad_w = 0.0
        grad_b = 0.0
        for x, y in data:
            error = (w * x + b) - y
            grad_w += error * x
            grad_b += error
        grad_w = (2.0 / n) * grad_w
        grad_b = (2.0 / n) * grad_b

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        mse = mean_squared_error((w, b), data)
        if run is not None:
            run.log_metric("train_mse", epoch, float(mse))
        else:
            print(f"epoch {epoch:03d} mse={mse:.4f}")

    return w, b


def dump_model_artifact(model: Tuple[float, float], run) -> None:
    if run is None:
        return
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump({"weight": model[0], "bias": model[1]}, tmp, indent=2)
            tmp.flush()
            tmp_path = tmp.name
        run.log_artifact("linear_model.json", tmp_path, kind="model", inline=True)
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def run_training(run, argv: Sequence[str]) -> Tuple[float, float]:
    args = parse_args(argv)
    random.seed(args.seed)
    data = make_regression_data(args.samples, args.noise)
    if run is not None:
        run.log_param("learning_rate", args.learning_rate)
        run.log_param("epochs", args.epochs)
        run.log_param("samples", args.samples)
        run.log_param("noise", args.noise)
        run.log_tag("model_type", "linear_regression")
        run.log_tag("framework", "pure_python")
    model = train_linear_model(data, args.epochs, args.learning_rate, run)
    mse = mean_squared_error(model, data)
    mae = mean_absolute_error(model, data)
    if run is not None:
        run.log_metric("final_mse", args.epochs, float(mse))
        run.log_metric("final_mae", args.epochs, float(mae))
    else:
        print(f"final mse={mse:.4f} mae={mae:.4f} model={model}")
    dump_model_artifact(model, run)
    return model


def waddle_main(run, argv: Sequence[str] | None = None) -> Tuple[float, float]:
    if argv is None:
        argv = []
    return run_training(run, argv)


def main() -> None:
    waddle_main(None, sys.argv[1:])


if __name__ == "__main__":
    main()
