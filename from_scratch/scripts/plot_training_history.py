"""Plot training loss and validation metrics from a saved history JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = ROOT / "results" / "training" / "training_history.json"


def plot_history(history_path: Path, output_path: Path | None = None, show: bool = False) -> Path:
    with history_path.open(encoding="utf-8") as file_handle:
        history = json.load(file_handle)

    losses = history.get("total_losses", [])
    if not losses:
        raise ValueError(f"No per-step losses found in {history_path}")

    output_path = output_path or history_path.with_name("training_loss.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    loss_steps = range(len(losses))
    axes[0].plot(loss_steps, losses, linewidth=0.8, label="training total loss")
    axes[0].set_ylabel("Training loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    verification_steps = history.get("verification_steps", [])
    verification_errors = history.get("verification_errors", [])
    if verification_steps and verification_errors:
        axes[1].plot(verification_steps, verification_errors, marker="o", label="verification angular error")
    iterations = history.get("iterations", [])
    evaluation_errors = history.get("overall_errors", [])
    if iterations and evaluation_errors:
        axes[1].plot(iterations, evaluation_errors, marker=".", label="logged angular error")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Angular error (rad)")
    axes[1].grid(alpha=0.25)
    if verification_steps or iterations:
        axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", nargs="?", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true", help="Open the plot window after saving")
    args = parser.parse_args()

    output = plot_history(args.history, args.output, args.show)
    print(f"Saved loss plot to {output}")


if __name__ == "__main__":
    main()