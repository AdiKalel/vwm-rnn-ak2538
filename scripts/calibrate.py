"""Print and save the derived calibration for one experiment config."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from vwm_scratch.calibration import derive_calibration, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "calibration_reports")
    args = parser.parse_args()

    config = load_config(args.config)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "name": config["name"],
        "model": config["model"],
        "experiment": config["experiment"],
        "derived": derive_calibration(config),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config['name']}.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved calibration report to {output_path}")


if __name__ == "__main__":
    main()
