"""Run the spin-model measurement snapshot benchmark."""

import argparse
from pathlib import Path

from bmera.experiments.snapshot_suite import (
    build_snapshot_dataset_suite,
    print_snapshot_comparison,
    run_snapshot_benchmark_suite,
)
from bmera.experiments.snapshots import SNAPSHOT_MODELS, save_benchmark_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "screen", "confirmation"), default="smoke")
    parser.add_argument("--model", action="append", choices=SNAPSHOT_MODELS)
    parser.add_argument("--seed", type=int, default=20260611, help="Dataset generation seed")
    parser.add_argument("--output", type=Path, default=Path("quantum_snapshot_runs/smoke"))
    args = parser.parse_args()
    datasets = build_snapshot_dataset_suite(
        args.profile, args.model or ("tfim",), cache_dir=args.output / "cache", seed=args.seed
    )
    results, ranking = run_snapshot_benchmark_suite(datasets, smoke=args.profile == "smoke")
    save_benchmark_results(results, ranking, args.output)
    print_snapshot_comparison(results, ranking)


if __name__ == "__main__":
    main()
