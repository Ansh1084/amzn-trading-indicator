"""Run the reproducible 10-minute ML challenger experiments on local data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.experiments.pipeline import DEFAULT_DATA, DEFAULT_OUTPUT, run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_experiments(args.data, args.output)
    print(f"Outputs: {result['output_dir']}")
    print("Validation selection:")
    print(result["ranking"].head(10).to_string(index=False))
    print("Untouched final test:")
    print(result["final_test"])


if __name__ == "__main__":
    main()
