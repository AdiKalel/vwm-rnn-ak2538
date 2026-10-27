"""Plot training loss and validation metrics from a saved history JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE_DIR = ROOT / "results" / "training_data"
LEGACY_HISTORY = ROOT / "results" / "training" / "training_history.json"


def plot_history(history_path: Path, output_path: Path | None = None, show: bool = False) -> Path:
    history_path = Path(history_path)
    if history_path.is_dir():
        history_path = history_path / "training_history.json"
    if history_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(history_path) as archive:
            history = json.loads(archive.read("training_history.json"))
        default_output = history_path.with_suffix(".png")
    else:
        with history_path.open(encoding="utf-8") as file_handle:
            history = json.load(file_handle)
        default_output = history_path.with_name("training_summary.png")

    output_path = Path(output_path) if output_path else default_output
    sys.path.insert(0, str(ROOT))
    from run_experiment import _plot_training

    return _plot_training(history, output_path, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true", help="Open the plot window after saving")
    args = parser.parse_args()

    history_path = args.history
    if history_path is None:
        runs = sorted(
            (path for path in ARCHIVE_DIR.glob("training_*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
        )
        archives = sorted(ARCHIVE_DIR.glob("training_*.zip"), key=lambda path: path.stat().st_mtime)
        history_path = runs[-1] if runs else (
            archives[-1] if archives else LEGACY_HISTORY
        )
    output = plot_history(history_path, args.output, args.show)
    print(f"Saved loss plot to {output}")


if __name__ == "__main__":
    main()