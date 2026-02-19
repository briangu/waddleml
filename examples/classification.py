"""Binary classification with a perceptron.

Generates two clusters and trains a single neuron with sigmoid.
Logs accuracy, loss, and learned parameters per epoch.

Usage:
    python examples/classification.py
    python examples/classification.py --epochs 200 --lr 0.1
    waddle serve
"""

import argparse
import math
import random
import waddle


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, z))))


def generate_clusters(n_per_class: int, separation: float):
    """Two Gaussian blobs."""
    data = []
    for _ in range(n_per_class):
        data.append((random.gauss(-separation, 1), random.gauss(-separation, 1), 0))
    for _ in range(n_per_class):
        data.append((random.gauss(separation, 1), random.gauss(separation, 1), 1))
    random.shuffle(data)
    return data


def train(data, epochs, lr):
    w1 = w2 = b = 0.0
    n = len(data)

    for epoch in range(epochs):
        total_loss = correct = 0
        gw1 = gw2 = gb = 0.0

        for x1, x2, y in data:
            pred = sigmoid(w1 * x1 + w2 * x2 + b)
            eps = 1e-15
            total_loss -= y * math.log(pred + eps) + (1 - y) * math.log(1 - pred + eps)
            correct += int((pred >= 0.5) == (y == 1))
            err = pred - y
            gw1 += err * x1
            gw2 += err * x2
            gb += err

        w1 -= lr * gw1 / n
        w2 -= lr * gw2 / n
        b -= lr * gb / n

        waddle.log({
            "train/loss": total_loss / n,
            "train/accuracy": correct / n,
            "params/w1": w1, "params/w2": w2, "params/bias": b,
        })

    return w1, w2, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--separation", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    with waddle.init(
        project="classification",
        name=f"perceptron_sep={args.separation}",
        config={"lr": args.lr, "epochs": args.epochs, "samples_per_class": args.samples_per_class, "separation": args.separation},
        tags={"model": "perceptron", "task": "binary_classification"},
    ):
        data = generate_clusters(args.samples_per_class, args.separation)
        w1, w2, b = train(data, args.epochs, args.lr)

        correct = sum(1 for x1, x2, y in data if (sigmoid(w1*x1 + w2*x2 + b) >= 0.5) == (y == 1))
        final_acc = correct / len(data)
        waddle.log({"eval/accuracy": final_acc})
        waddle.log_param("final_accuracy", round(final_acc, 4))
        print(f"Final accuracy: {final_acc:.1%}")


if __name__ == "__main__":
    main()
