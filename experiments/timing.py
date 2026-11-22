"""Record synchronized BMERA timings as JSON."""

import argparse
import json

from bmera.benchmarking import benchmark_training_scaling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--engine", choices=("autodiff", "cached"), default="autodiff")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rows = benchmark_training_scaling(
        args.qubits, batch_sizes=tuple(args.batch_size), repeats=args.repeats,
        device=args.device, grad_engine=args.engine,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
