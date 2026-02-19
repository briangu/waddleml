"""Minimal WaddleML example — works anywhere, no git required.

Usage:
    python examples/quickstart.py
    waddle ls            # see your run in the terminal
    waddle serve         # open http://127.0.0.1:8080
"""

import random
import waddle

with waddle.init(project="quickstart", config={"lr": 0.01}):
    loss = 1.0
    for step in range(50):
        loss *= random.uniform(0.9, 0.99)
        waddle.log({"loss": loss})

    waddle.log_param("final_loss", loss)
    waddle.log_tag("status", "demo")
    print(f"Done! final loss={loss:.4f}")
