# BMERA Simulator

A JAX tensor-network simulator for branching MERA circuits. It computes single-qubit Pauli-Z expectations for angle-encoded product states and supports JAX autodiff, cached gradients, and microbatching.

## Scientific background

The simulator builds on the branching MERA construction, and its classification experiments draw on work applying that construction to machine learning.

- G. Evenbly and G. Vidal introduced branching MERA and showed how to compute local observables efficiently. *Class of Highly Entangled Many-Body States that can be Efficiently Simulated*, Physical Review Letters **112**, 240502 (2014). [DOI](https://doi.org/10.1103/PhysRevLett.112.240502)
- Y.-Y. Hou, J. Li, T. Xu, and X.-Y. Liu developed a hybrid quantum-classical classifier using branching MERA. *A hybrid quantum-classical classification model based on branching multi-scale entanglement renormalization ansatz*, Scientific Reports **14**, 18521 (2024). [DOI](https://doi.org/10.1038/s41598-024-69384-6)

## Quick start

Use Python 3.12 or later. Python 3.12 is the version tested in CI.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/quickstart.py
```

The example runs on CPU and prints the four Pauli-Z expectations. Each block applies RX and RZ on both wires and finishes with CNOT (type 1), CRX (type 2), or CRZ (type 3). The lower wire controls the upper wire in all three cases.

`build_bmera_meta(n_qubits, type)` accepts powers of two starting at 2. Parameters have shape `(depth, n_qubits, n_per_block)` or `(depth * n_qubits, n_per_block)`, with `depth = log2(n_qubits)` and four parameters for CNOT blocks or five for the others. An input vector has one angle per qubit. For batched inputs of shape `(batch_size, n_qubits)`, `forward_batch` returns an array of the same shape.

## Notebooks and tests

```sh
python -m pip install -e '.[notebooks,reference,experiments]'
JAX_PLATFORMS=cpu python -m unittest discover -v
./start_notebook.sh
```

The launcher opens the [CPU demonstration](notebooks/BMERA_simulator.ipynb). The [validation notebook](notebooks/BMERA_validation.ipynb) compares the implementations, and the [research notebook](notebooks/BMERA_simulator_exec.ipynb) provides opt-in switches for longer experiments. The optional reference backends are PennyLane and [TensorCircuit-NG](https://tensorcircuit-ng.readthedocs.io/en/latest/faq.html).

## Experiments

```sh
JAX_PLATFORMS=cpu python experiments/timing.py > timings.json
JAX_PLATFORMS=cpu python experiments/snapshots.py --output quantum_snapshot_runs/smoke
```

The timing script saves first-call and steady-state timings with the device, dtype, versions, batches, and repeat counts. The snapshot script defaults to a small TFIM run. Use `--profile screen` or `--profile confirmation` for larger runs and `--model` to select spin models. The matched MLP budget uses coefficients from blocks in the contraction plan plus the linear head. The output also reports how many coefficients the BMERA arrays allocate. The classical comparator is selected by validation accuracy before comparison on the near-critical test split.

The [RTX 5070 report](docs/reports/rtx5070/bmera_rtx5070_theoretical_estimate.tex) records historical type-1 GPU measurements. The [saved 256-qubit sweep](docs/reports/rtx5070/historical_256_qubit_sweep.json) contains the original per-batch rows. The report identifies estimates read from older plots. Timing runs on the current package produce a new series.
