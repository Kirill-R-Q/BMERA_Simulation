"""Quantum-spin measurement snapshot datasets and BMERA/classical benchmarks.

The classifiers consume classical finite-shot measurement records. They do not
consume entangled wavefunctions directly.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from bmera.topology import active_block_indices, build_bmera_meta


SNAPSHOT_MODELS = ("tfim", "cluster_ising", "xxz")
SNAPSHOT_SPLITS = ("train", "val", "iid_test", "critical_test")
PHASE_RANGES = {
    "train": {-1.0: (0.25, 0.75), 1.0: (1.35, 1.85)},
    "val": {-1.0: (0.78, 0.90), 1.0: (1.10, 1.30)},
    "iid_test": {-1.0: (0.25, 0.75), 1.0: (1.35, 1.85)},
    "critical_test": {-1.0: (0.92, 0.99), 1.0: (1.01, 1.08)},
}
SCREEN_STATE_COUNTS = {
    "train": 10,
    "val": 4,
    "iid_test": 6,
    "critical_test": 6,
}
CONFIRM_STATE_COUNTS = {
    "train": 20,
    "val": 8,
    "iid_test": 12,
    "critical_test": 12,
}
MODEL_SEED_OFFSETS = {
    "tfim": 11003,
    "cluster_ising": 22003,
    "xxz": 33013,
}


def _validate_model(model: str) -> str:
    model = str(model).lower()
    if model not in SNAPSHOT_MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {SNAPSHOT_MODELS}.")
    return model


def _validate_n_qubits(n_qubits: int) -> int:
    n_qubits = int(n_qubits)
    if n_qubits < 4 or n_qubits % 2:
        raise ValueError("Snapshot datasets require an even number of qubits >= 4.")
    return n_qubits


def _bit_z(states: np.ndarray, qubit: int) -> np.ndarray:
    return 1.0 - 2.0 * ((states >> int(qubit)) & 1)


@lru_cache(maxsize=None)
def hamiltonian_parts(model: str, n_qubits: int) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Return H(parameter) = H0 + parameter * H1 in the computational basis."""
    model = _validate_model(model)
    n_qubits = _validate_n_qubits(n_qubits)
    dim = 1 << n_qubits
    states = np.arange(dim, dtype=np.int64)

    if model == "tfim":
        diagonal = np.zeros(dim, dtype=np.float64)
        for q in range(n_qubits):
            diagonal -= _bit_z(states, q) * _bit_z(states, (q + 1) % n_qubits)
        h0 = sparse.diags(diagonal, format="csr")
        rows = np.tile(states, n_qubits)
        cols = np.concatenate([states ^ (1 << q) for q in range(n_qubits)])
        data = -np.ones(rows.size, dtype=np.float64)
        h1 = sparse.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
        return h0, h1

    if model == "cluster_ising":
        rows = []
        cols = []
        data = []
        for q in range(n_qubits):
            left = (q - 1) % n_qubits
            right = (q + 1) % n_qubits
            rows.append(states ^ (1 << q))
            cols.append(states)
            data.append(-_bit_z(states, left) * _bit_z(states, right))
        h0 = sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(dim, dim),
        ).tocsr()
        rows_x = np.tile(states, n_qubits)
        cols_x = np.concatenate([states ^ (1 << q) for q in range(n_qubits)])
        h1 = sparse.coo_matrix(
            (-np.ones(rows_x.size), (rows_x, cols_x)),
            shape=(dim, dim),
        ).tocsr()
        return h0, h1

    rows = []
    cols = []
    data = []
    zz_diagonal = np.zeros(dim, dtype=np.float64)
    for q in range(n_qubits):
        r = (q + 1) % n_qubits
        zq = _bit_z(states, q)
        zr = _bit_z(states, r)
        zz_diagonal += zq * zr
        opposite = zq != zr
        source = states[opposite]
        rows.append(source ^ (1 << q) ^ (1 << r))
        cols.append(source)
        data.append(np.full(source.size, 2.0, dtype=np.float64))
    h0 = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
    ).tocsr()
    h1 = sparse.diags(zz_diagonal, format="csr")
    return h0, h1


def build_hamiltonian(model: str, n_qubits: int, parameter: float) -> sparse.csr_matrix:
    h0, h1 = hamiltonian_parts(model, n_qubits)
    return (h0 + float(parameter) * h1).tocsr()


def cache_hamiltonian_parts(model: str, n_qubits: int, cache_dir: str | Path) -> Dict[str, str]:
    model = _validate_model(model)
    n_qubits = _validate_n_qubits(n_qubits)
    cache_dir = Path(cache_dir) / "operators"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "h0": cache_dir / f"{model}_n{n_qubits}_h0.npz",
        "h1": cache_dir / f"{model}_n{n_qubits}_h1.npz",
    }
    h0, h1 = hamiltonian_parts(model, n_qubits)
    if not paths["h0"].exists():
        sparse.save_npz(paths["h0"], h0)
    if not paths["h1"].exists():
        sparse.save_npz(paths["h1"], h1)
    return {name: str(path) for name, path in paths.items()}


def _state_cache_path(
    model: str,
    n_qubits: int,
    parameter: float,
    cache_dir: str | Path,
) -> Path:
    key = f"{model}|{n_qubits}|{float(parameter):.14g}".encode("ascii")
    digest = hashlib.sha256(key).hexdigest()[:16]
    return Path(cache_dir) / "ground_states" / f"{model}_n{n_qubits}_{digest}.npz"


def ground_state(
    model: str,
    n_qubits: int,
    parameter: float,
    *,
    cache_dir: str | Path | None = None,
    tol: float = 1e-10,
    maxiter: int | None = None,
) -> Tuple[float, np.ndarray, float]:
    """Compute or load a normalized ground state and its residual norm."""
    model = _validate_model(model)
    n_qubits = _validate_n_qubits(n_qubits)
    cache_path = None
    if cache_dir is not None:
        cache_hamiltonian_parts(model, n_qubits, cache_dir)
        cache_path = _state_cache_path(model, n_qubits, parameter, cache_dir)
        if cache_path.exists():
            saved = np.load(cache_path)
            return (
                float(saved["energy"]),
                np.asarray(saved["state"], dtype=np.complex128),
                float(saved["residual"]),
            )

    hamiltonian = build_hamiltonian(model, n_qubits, parameter)
    dim = hamiltonian.shape[0]
    if dim <= 256:
        values, vectors = np.linalg.eigh(hamiltonian.toarray())
        energy = float(values[0])
        state = np.asarray(vectors[:, 0], dtype=np.complex128)
    else:
        seed = MODEL_SEED_OFFSETS[model] + 1009 * n_qubits + int(round(float(parameter) * 1e6))
        rng = np.random.default_rng(seed)
        v0 = rng.normal(size=dim)
        values, vectors = eigsh(
            hamiltonian,
            k=1,
            which="SA",
            v0=v0,
            tol=float(tol),
            maxiter=maxiter,
        )
        energy = float(values[0])
        state = np.asarray(vectors[:, 0], dtype=np.complex128)
    state /= np.linalg.norm(state)
    residual = float(np.linalg.norm(hamiltonian @ state - energy * state))
    if residual > max(1e-7, 100.0 * float(tol)):
        raise RuntimeError(
            f"Ground-state residual too large for {model}, n={n_qubits}, "
            f"parameter={parameter}: {residual:.3e}"
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            model=np.asarray(model),
            n_qubits=np.asarray(n_qubits),
            parameter=np.asarray(parameter),
            energy=np.asarray(energy),
            state=state,
            residual=np.asarray(residual),
        )
    return energy, state, residual


def pauli_xz_expectation(
    state: np.ndarray,
    n_qubits: int,
    x_qubits: Iterable[int] = (),
    z_qubits: Iterable[int] = (),
) -> float:
    """Expectation of a Pauli string containing only X and Z operators."""
    state = np.asarray(state, dtype=np.complex128)
    states = np.arange(state.size, dtype=np.int64)
    flip_mask = 0
    for qubit in x_qubits:
        flip_mask ^= 1 << int(qubit)
    phase = np.ones(state.size, dtype=np.float64)
    for qubit in z_qubits:
        phase *= _bit_z(states, int(qubit))
    value = np.vdot(state[states ^ flip_mask], phase * state)
    return float(np.real_if_close(value))


def physical_observables(model: str, state: np.ndarray, n_qubits: int) -> Dict[str, float]:
    model = _validate_model(model)
    n_qubits = _validate_n_qubits(n_qubits)
    zz = np.mean(
        [
            pauli_xz_expectation(state, n_qubits, z_qubits=(q, (q + 1) % n_qubits))
            for q in range(n_qubits)
        ]
    )
    x = np.mean(
        [pauli_xz_expectation(state, n_qubits, x_qubits=(q,)) for q in range(n_qubits)]
    )
    cluster = np.mean(
        [
            pauli_xz_expectation(
                state,
                n_qubits,
                x_qubits=(q,),
                z_qubits=((q - 1) % n_qubits, (q + 1) % n_qubits),
            )
            for q in range(n_qubits)
        ]
    )
    staggered_zz = np.mean(
        [
            ((-1.0) ** (r - q))
            * pauli_xz_expectation(state, n_qubits, z_qubits=(q, r))
            for q in range(n_qubits)
            for r in range(n_qubits)
        ]
    )
    return {
        "mean_nn_zz": float(zz),
        "mean_x": float(x),
        "mean_cluster": float(cluster),
        "staggered_structure_factor": float(staggered_zz),
    }


def _largest_even_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(int(value), int(limit)), 1, -1):
        if candidate % 2 == 0 and value % candidate == 0:
            return candidate
    if value % 2:
        raise ValueError("The number of snapshots must be even.")
    return 2


def balanced_basis_masks(
    n_qubits: int,
    n_snapshots: int,
    seed: int,
    *,
    max_bank_size: int = 32,
) -> np.ndarray:
    """Return a repeated X/Z mask bank balanced at every qubit."""
    n_qubits = _validate_n_qubits(n_qubits)
    n_snapshots = int(n_snapshots)
    if n_snapshots <= 0 or n_snapshots % 2:
        raise ValueError("n_snapshots must be a positive even integer.")
    bank_size = _largest_even_divisor_at_most(n_snapshots, max_bank_size)
    rng = np.random.default_rng(int(seed))
    half = rng.integers(0, 2, size=(bank_size // 2, n_qubits), dtype=np.uint8)
    bank = np.concatenate([half, 1 - half], axis=0)
    bank = bank[rng.permutation(bank_size)]
    masks = np.tile(bank, (n_snapshots // bank_size, 1))
    masks = masks[rng.permutation(n_snapshots)]
    if not np.all(masks.sum(axis=0) * 2 == n_snapshots):
        raise RuntimeError("Basis-mask construction failed to balance X and Z measurements.")
    return masks


def _apply_hadamards(state: np.ndarray, basis_mask: np.ndarray) -> np.ndarray:
    n_qubits = int(basis_mask.size)
    tensor = np.asarray(state, dtype=np.complex128).reshape((2,) * n_qubits)
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    hadamard = np.asarray([[1.0, 1.0], [1.0, -1.0]]) * inv_sqrt_two
    for qubit in np.flatnonzero(basis_mask):
        axis = n_qubits - 1 - int(qubit)
        tensor = np.tensordot(hadamard, tensor, axes=((1,), (axis,)))
        tensor = np.moveaxis(tensor, 0, axis)
    return tensor.reshape(-1)


def sample_measurement_snapshots(
    state: np.ndarray,
    basis_masks: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Sample +/-1 outcomes for local X/Z basis masks."""
    state = np.asarray(state, dtype=np.complex128)
    basis_masks = np.asarray(basis_masks, dtype=np.uint8)
    if basis_masks.ndim != 2:
        raise ValueError("basis_masks must have shape (snapshots, qubits).")
    n_snapshots, n_qubits = basis_masks.shape
    if state.shape != (1 << n_qubits,):
        raise ValueError(f"State shape {state.shape} does not match {n_qubits} qubits.")
    rng = np.random.default_rng(int(seed))
    outcomes = np.empty((n_snapshots, n_qubits), dtype=np.int8)
    unique_masks, inverse = np.unique(basis_masks, axis=0, return_inverse=True)
    bit_positions = np.arange(n_qubits, dtype=np.int64)
    for mask_index, mask in enumerate(unique_masks):
        row_indices = np.flatnonzero(inverse == mask_index)
        transformed = _apply_hadamards(state, mask)
        probabilities = np.abs(transformed) ** 2
        probabilities /= probabilities.sum()
        samples = rng.choice(state.size, size=row_indices.size, p=probabilities)
        bits = ((samples[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
        outcomes[row_indices] = 1 - 2 * bits
    return outcomes


def encode_snapshot_angles(basis_masks: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    basis_masks = np.asarray(basis_masks, dtype=np.uint8)
    outcomes = np.asarray(outcomes, dtype=np.int8)
    if basis_masks.shape != outcomes.shape:
        raise ValueError("basis_masks and outcomes must have identical shapes.")
    if not np.all(np.isin(outcomes, (-1, 1))):
        raise ValueError("Measurement outcomes must be +/-1.")
    z_angles = (1.0 - outcomes.astype(np.float32)) * (np.pi / 4.0)
    x_angles = outcomes.astype(np.float32) * (np.pi / 4.0)
    return np.where(basis_masks == 0, z_angles, x_angles).astype(np.float32)


def _sample_parameters(
    interval: Tuple[float, float],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = rng.uniform(float(interval[0]), float(interval[1]), size=int(count))
    return np.sort(values.astype(np.float64))


def _dataset_cache_key(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _flatten_dataset(dataset: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(dataset["metadata"], sort_keys=True))
    }
    for split_name in SNAPSHOT_SPLITS:
        split = dataset[split_name]
        for key, value in split.items():
            arrays[f"{split_name}__{key}"] = np.asarray(value)
    return arrays


def _unflatten_dataset(saved: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    metadata = json.loads(str(saved["metadata_json"].item()))
    dataset: Dict[str, Any] = {"metadata": metadata}
    for split_name in SNAPSHOT_SPLITS:
        prefix = f"{split_name}__"
        dataset[split_name] = {
            key[len(prefix):]: np.asarray(saved[key])
            for key in saved.files
            if key.startswith(prefix)
        }
    return dataset


def generate_snapshot_dataset(
    model: str,
    n_qubits: int,
    *,
    state_counts: Mapping[str, int] = SCREEN_STATE_COUNTS,
    snapshots_per_state: int = 128,
    seed: int = 20260611,
    cache_dir: str | Path = "quantum_snapshot_runs/cache",
    force: bool = False,
) -> Dict[str, Any]:
    """Generate state-disjoint train/validation/IID/near-critical snapshots."""
    model = _validate_model(model)
    n_qubits = _validate_n_qubits(n_qubits)
    state_counts = {name: int(state_counts[name]) for name in SNAPSHOT_SPLITS}
    config = {
        "version": 1,
        "model": model,
        "n_qubits": n_qubits,
        "state_counts_per_class": state_counts,
        "snapshots_per_state": int(snapshots_per_state),
        "seed": int(seed),
        "phase_ranges": PHASE_RANGES,
    }
    cache_dir = Path(cache_dir)
    dataset_path = cache_dir / "datasets" / f"{model}_n{n_qubits}_{_dataset_cache_key(config)}.npz"
    if dataset_path.exists() and not force:
        with np.load(dataset_path, allow_pickle=False) as saved:
            dataset = _unflatten_dataset(saved)
        validate_snapshot_dataset(dataset)
        return dataset

    model_seed = int(seed) + MODEL_SEED_OFFSETS[model] + 1009 * n_qubits
    rng = np.random.default_rng(model_seed)
    dataset: Dict[str, Any] = {
        "metadata": {
            **config,
            "dataset_path": str(dataset_path),
            "operator_paths": cache_hamiltonian_parts(model, n_qubits, cache_dir),
            "description": (
                "Classical finite-shot X/Z measurement records sampled from exact "
                "ground states; no wavefunction is supplied to BMERA."
            ),
        }
    }
    all_state_ids = set()
    for split_index, split_name in enumerate(SNAPSHOT_SPLITS):
        split_parts: Dict[str, list[np.ndarray]] = {
            "X": [],
            "y": [],
            "state_id": [],
            "parameter": [],
            "basis": [],
            "outcomes": [],
            "energy": [],
            "residual": [],
        }
        split_state_counter = 0
        for label in (-1.0, 1.0):
            values = _sample_parameters(
                PHASE_RANGES[split_name][label],
                state_counts[split_name],
                rng,
            )
            for parameter in values:
                state_id = (
                    f"{model}:n{n_qubits}:{split_name}:{int(label):+d}:"
                    f"{split_state_counter:04d}:{parameter:.12f}"
                )
                split_state_counter += 1
                if state_id in all_state_ids:
                    raise RuntimeError(f"Duplicate generated state ID: {state_id}")
                all_state_ids.add(state_id)
                energy, state, residual = ground_state(
                    model,
                    n_qubits,
                    float(parameter),
                    cache_dir=cache_dir,
                )
                sample_seed = (
                    model_seed
                    + 1000003 * split_index
                    + 10007 * split_state_counter
                    + (0 if label < 0 else 503)
                )
                basis = balanced_basis_masks(
                    n_qubits,
                    int(snapshots_per_state),
                    sample_seed,
                )
                outcomes = sample_measurement_snapshots(state, basis, sample_seed + 1)
                angles = encode_snapshot_angles(basis, outcomes)
                count = int(snapshots_per_state)
                split_parts["X"].append(angles)
                split_parts["y"].append(np.full(count, label, dtype=np.float32))
                split_parts["state_id"].append(np.full(count, state_id, dtype=f"<U{len(state_id)}"))
                split_parts["parameter"].append(np.full(count, parameter, dtype=np.float64))
                split_parts["basis"].append(basis.astype(np.uint8))
                split_parts["outcomes"].append(outcomes.astype(np.int8))
                split_parts["energy"].append(np.full(count, energy, dtype=np.float64))
                split_parts["residual"].append(np.full(count, residual, dtype=np.float64))
        dataset[split_name] = {
            key: np.concatenate(parts, axis=0)
            for key, parts in split_parts.items()
        }

    validate_snapshot_dataset(dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **_flatten_dataset(dataset))
    return dataset


def validate_snapshot_dataset(dataset: Mapping[str, Any]) -> None:
    metadata = dataset["metadata"]
    n_qubits = int(metadata["n_qubits"])
    snapshots_per_state = int(metadata["snapshots_per_state"])
    state_sets = {}
    for split_name in SNAPSHOT_SPLITS:
        split = dataset[split_name]
        size = int(np.asarray(split["y"]).size)
        required = ("X", "y", "state_id", "parameter", "basis", "outcomes", "energy", "residual")
        if any(key not in split for key in required):
            raise ValueError(f"Missing fields in split {split_name}.")
        if np.asarray(split["X"]).shape != (size, n_qubits):
            raise ValueError(f"Unexpected X shape in {split_name}: {np.asarray(split['X']).shape}")
        if np.asarray(split["basis"]).shape != (size, n_qubits):
            raise ValueError(f"Unexpected basis shape in {split_name}.")
        if np.asarray(split["outcomes"]).shape != (size, n_qubits):
            raise ValueError(f"Unexpected outcomes shape in {split_name}.")
        if not np.all(np.isfinite(split["X"])):
            raise ValueError(f"Non-finite encoded snapshots in {split_name}.")
        if set(np.unique(split["y"]).tolist()) != {-1.0, 1.0}:
            raise ValueError(f"Split {split_name} is not binary and balanced by construction.")
        if not np.all(np.isin(split["basis"], (0, 1))):
            raise ValueError(f"Invalid basis values in {split_name}.")
        if not np.all(np.isin(split["outcomes"], (-1, 1))):
            raise ValueError(f"Invalid measurement outcomes in {split_name}.")
        states, counts = np.unique(split["state_id"], return_counts=True)
        if not np.all(counts == snapshots_per_state):
            raise ValueError(f"State snapshot counts are inconsistent in {split_name}.")
        state_sets[split_name] = set(states.tolist())
        for state_id in states:
            rows = np.flatnonzero(split["state_id"] == state_id)
            if not np.all(np.sum(split["basis"][rows], axis=0) * 2 == rows.size):
                raise ValueError(f"Basis masks are not balanced for state {state_id}.")
    for left_index, left in enumerate(SNAPSHOT_SPLITS):
        for right in SNAPSHOT_SPLITS[left_index + 1:]:
            overlap = state_sets[left] & state_sets[right]
            if overlap:
                raise ValueError(f"State leakage between {left} and {right}: {sorted(overlap)[:2]}")


def grouped_state_scores(
    scores: np.ndarray,
    split: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    state_ids = np.asarray(split["state_id"])
    labels = np.asarray(split["y"], dtype=np.float64)
    if scores.size != labels.size:
        raise ValueError("Prediction count does not match split size.")
    unique_ids = np.unique(state_ids)
    state_scores = []
    state_labels = []
    state_parameters = []
    for state_id in unique_ids:
        rows = state_ids == state_id
        unique_labels = np.unique(labels[rows])
        if unique_labels.size != 1:
            raise ValueError(f"State {state_id} has inconsistent labels.")
        state_scores.append(float(np.mean(scores[rows])))
        state_labels.append(float(unique_labels[0]))
        state_parameters.append(float(np.mean(np.asarray(split["parameter"])[rows])))
    return {
        "state_id": unique_ids,
        "score": np.asarray(state_scores, dtype=np.float64),
        "y": np.asarray(state_labels, dtype=np.float64),
        "parameter": np.asarray(state_parameters, dtype=np.float64),
    }


def balanced_accuracy(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    predicted = np.where(np.asarray(scores) >= 0.0, 1.0, -1.0)
    recalls = [
        float(np.mean(predicted[y_true == label] == label))
        for label in (-1.0, 1.0)
        if np.any(y_true == label)
    ]
    return float(np.mean(recalls))


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[y_true > 0]
    negative = scores[y_true < 0]
    if not positive.size or not negative.size:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))


def evaluate_snapshot_scores(
    scores: np.ndarray,
    split: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(split["y"], dtype=np.float64)
    grouped = grouped_state_scores(scores, split)
    return {
        "snapshot_balanced_accuracy": balanced_accuracy(y, scores),
        "snapshot_auroc": binary_auroc(y, scores),
        "state_balanced_accuracy": balanced_accuracy(grouped["y"], grouped["score"]),
        "state_auroc": binary_auroc(grouped["y"], grouped["score"]),
        "state_predictions": grouped,
    }


def _require_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise ImportError(
            "Classical snapshot baselines require scikit-learn. "
            "Install the versions listed in requirements.txt."
        ) from exc
    return {
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "MLPClassifier": MLPClassifier,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
    }


def _classical_candidates(model_name: str, n_qubits: int, bmera_parameter_count: int):
    hidden = max(1, int(round((int(bmera_parameter_count) - 1) / (int(n_qubits) + 2))))
    if model_name == "logistic":
        return [{"C": value} for value in (0.1, 1.0, 10.0)]
    if model_name == "rbf_svm":
        return [
            {"C": c_value, "gamma": gamma}
            for c_value in (1.0, 10.0)
            for gamma in ("scale", 0.1, 1.0)
        ]
    if model_name == "random_forest":
        return [
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth in (None, 8)
            for leaf in (1, 4)
        ]
    if model_name == "matched_mlp":
        return [
            {"hidden_layer_sizes": (hidden,), "alpha": alpha, "learning_rate_init": lr}
            for alpha in (1e-4, 1e-2)
            for lr in (1e-3, 1e-2)
        ]
    raise ValueError(f"Unknown classical model {model_name!r}.")


def _build_classical_estimator(model_name: str, config: Mapping[str, Any], seed: int):
    sk = _require_sklearn()
    if model_name == "logistic":
        estimator = sk["LogisticRegression"](
            C=float(config["C"]),
            max_iter=2000,
            random_state=int(seed),
        )
        return sk["make_pipeline"](sk["StandardScaler"](), estimator)
    if model_name == "rbf_svm":
        estimator = sk["SVC"](
            C=float(config["C"]),
            gamma=config["gamma"],
            kernel="rbf",
            random_state=int(seed),
        )
        return sk["make_pipeline"](sk["StandardScaler"](), estimator)
    if model_name == "random_forest":
        return sk["RandomForestClassifier"](
            n_estimators=300,
            max_depth=config["max_depth"],
            min_samples_leaf=int(config["min_samples_leaf"]),
            random_state=int(seed),
            n_jobs=-1,
        )
    return sk["make_pipeline"](
        sk["StandardScaler"](),
        sk["MLPClassifier"](
            hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
            alpha=float(config["alpha"]),
            learning_rate_init=float(config["learning_rate_init"]),
            max_iter=1000,
            # Hyperparameters are selected on the state-disjoint validation
            # split, so an additional snapshot-level split is both redundant
            # and invalid for very small smoke datasets.
            early_stopping=False,
            random_state=int(seed),
        ),
    )


def _estimator_scores(estimator, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=np.float64)
    probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    classes = np.asarray(estimator.classes_)
    if hasattr(estimator, "steps"):
        classes = np.asarray(estimator.steps[-1][1].classes_)
    positive_index = int(np.flatnonzero(classes > 0)[0])
    return 2.0 * probabilities[:, positive_index] - 1.0


def _fitted_parameter_count(estimator, model_name: str) -> int:
    fitted = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if model_name == "logistic":
        return int(fitted.coef_.size + fitted.intercept_.size)
    if model_name == "rbf_svm":
        return int(fitted.dual_coef_.size + fitted.intercept_.size)
    if model_name == "matched_mlp":
        return int(
            sum(weights.size for weights in fitted.coefs_)
            + sum(bias.size for bias in fitted.intercepts_)
        )
    return int(
        sum(tree.tree_.threshold.size + tree.tree_.value.size for tree in fitted.estimators_)
    )


def tune_classical_baseline(
    dataset: Mapping[str, Any],
    model_name: str,
    *,
    bmera_parameter_count: int,
    final_seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> Dict[str, Any]:
    train = dataset["train"]
    val = dataset["val"]
    best = None
    for config in _classical_candidates(
        model_name,
        int(dataset["metadata"]["n_qubits"]),
        int(bmera_parameter_count),
    ):
        estimator = _build_classical_estimator(model_name, config, seed=0)
        estimator.fit(train["X"], train["y"])
        val_scores = _estimator_scores(estimator, val["X"])
        metrics = evaluate_snapshot_scores(val_scores, val)
        key = (metrics["state_balanced_accuracy"], metrics["state_auroc"])
        if best is None or key > best["selection_key"]:
            best = {"config": dict(config), "selection_key": key}

    runs = []
    for seed in final_seeds:
        estimator = _build_classical_estimator(model_name, best["config"], int(seed))
        fit_start = time.perf_counter()
        estimator.fit(train["X"], train["y"])
        fit_seconds = time.perf_counter() - fit_start
        split_metrics = {}
        for split_name in ("val", "iid_test", "critical_test"):
            split = dataset[split_name]
            predict_start = time.perf_counter()
            scores = _estimator_scores(estimator, split["X"])
            inference_seconds = time.perf_counter() - predict_start
            metrics = evaluate_snapshot_scores(scores, split)
            split_metrics[split_name] = {
                **{key: value for key, value in metrics.items() if key != "state_predictions"},
                "inference_seconds": float(inference_seconds),
                "snapshot_scores": scores,
                "state_predictions": metrics["state_predictions"],
            }
        runs.append(
            {
                "seed": int(seed),
                "fit_seconds": float(fit_seconds),
                "parameter_count": _fitted_parameter_count(estimator, model_name),
                "splits": split_metrics,
            }
        )
    return {
        "model": model_name,
        "selected_config": best["config"],
        "validation_selection_key": best["selection_key"],
        "runs": runs,
    }


def _bmera_dependencies():
    import jax
    import jax.numpy as jnp

    from bmera import (
        forward_batch,
        loss_and_grad_autodiff_microbatched,
    )

    return jax, jnp, build_bmera_meta, forward_batch, loss_and_grad_autodiff_microbatched


def select_jax_device(prefer_gpu: bool = True):
    jax, _, _, _, _ = _bmera_dependencies()
    if prefer_gpu:
        gpu_devices = [
            device
            for device in jax.devices()
            if device.platform in ("gpu", "cuda", "rocm")
        ]
        if gpu_devices:
            return gpu_devices[0], False
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("No JAX device is available.")
    return cpu_devices[0], True


def bmera_parameter_count(n_qubits: int, block_type: int = 1, *, allocated: bool = False) -> int:
    """Count circuit coefficients plus the linear head, using active blocks by default."""
    meta = build_bmera_meta(n_qubits, block_type)
    n_rows = int(meta["depth"]) * n_qubits if allocated else len(active_block_indices(meta))
    return n_rows * int(meta["n_per_block"]) + n_qubits


def _init_bmera(n_qubits: int, seed: int, init_scale: float, block_type: int = 1):
    jax, jnp, get_meta, _, _ = _bmera_dependencies()
    del jax
    meta = get_meta(int(n_qubits), int(block_type))
    rng = np.random.default_rng(int(seed))
    shape = (int(meta["depth"]) * int(n_qubits), int(meta["n_per_block"]))
    params = jnp.asarray(rng.uniform(-init_scale, init_scale, size=shape), dtype=jnp.float32)
    head_scale = float(init_scale) / math.sqrt(int(n_qubits))
    head = jnp.asarray(rng.uniform(-head_scale, head_scale, size=n_qubits), dtype=jnp.float32)
    return meta, params, head


def _init_adam(params, head, learning_rate: float):
    jax, jnp, _, _, _ = _bmera_dependencies()
    del jax
    return {
        "step": 0,
        "learning_rate": float(learning_rate),
        "m_params": jnp.zeros_like(params),
        "v_params": jnp.zeros_like(params),
        "m_head": jnp.zeros_like(head),
        "v_head": jnp.zeros_like(head),
    }


def _adam_step(params, head, state, grad_params, grad_head):
    _, jnp, _, _, _ = _bmera_dependencies()
    step = int(state["step"]) + 1
    beta1 = jnp.asarray(0.9, dtype=params.dtype)
    beta2 = jnp.asarray(0.999, dtype=params.dtype)
    eps = jnp.asarray(1e-8, dtype=params.dtype)
    lr = jnp.asarray(state["learning_rate"], dtype=params.dtype)
    m_params = beta1 * state["m_params"] + (1.0 - beta1) * grad_params
    v_params = beta2 * state["v_params"] + (1.0 - beta2) * grad_params**2
    m_head = beta1 * state["m_head"] + (1.0 - beta1) * grad_head
    v_head = beta2 * state["v_head"] + (1.0 - beta2) * grad_head**2
    correction1 = 1.0 - beta1**step
    correction2 = 1.0 - beta2**step
    params = params - lr * (m_params / correction1) / (jnp.sqrt(v_params / correction2) + eps)
    head = head - lr * (m_head / correction1) / (jnp.sqrt(v_head / correction2) + eps)
    return params, head, {
        **state,
        "step": step,
        "m_params": m_params,
        "v_params": v_params,
        "m_head": m_head,
        "v_head": v_head,
    }


def _block_until_ready(tree) -> None:
    jax, _, _, _, _ = _bmera_dependencies()
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def bmera_predict(
    params,
    head,
    X: np.ndarray,
    meta: Mapping[str, Any],
    *,
    batch_size: int,
    device,
) -> np.ndarray:
    jax, jnp, _, forward_batch, _ = _bmera_dependencies()
    predictions = []
    for start in range(0, X.shape[0], int(batch_size)):
        batch = jax.device_put(
            jnp.asarray(X[start:start + int(batch_size)], dtype=jnp.float32),
            device,
        )
        scores = forward_batch(params, batch, meta) @ head
        scores.block_until_ready()
        predictions.append(np.asarray(jax.device_get(scores), dtype=np.float64))
    return np.concatenate(predictions)


def train_bmera_snapshot_once(
    dataset: Mapping[str, Any],
    *,
    seed: int,
    learning_rate: float,
    init_scale: float,
    max_steps: int = 300,
    batch_size: int = 128,
    eval_every: int = 10,
    patience: int = 8,
    block_type: int = 1,
    prefer_gpu: bool = True,
) -> Dict[str, Any]:
    jax, jnp, _, _, loss_and_grad = _bmera_dependencies()
    n_qubits = int(dataset["metadata"]["n_qubits"])
    device, used_cpu_fallback = select_jax_device(prefer_gpu=prefer_gpu)
    meta, params, head = _init_bmera(n_qubits, seed, init_scale, block_type)
    params = jax.device_put(params, device)
    head = jax.device_put(head, device)
    state = _init_adam(params, head, learning_rate)
    train = dataset["train"]
    rng = np.random.default_rng(int(seed) + 70001)
    best = None
    stale_evaluations = 0
    losses = []
    fit_start = time.perf_counter()
    for step in range(1, int(max_steps) + 1):
        indices = rng.choice(train["X"].shape[0], size=int(batch_size), replace=False)
        X_batch = jax.device_put(jnp.asarray(train["X"][indices], dtype=jnp.float32), device)
        y_batch = jax.device_put(jnp.asarray(train["y"][indices], dtype=jnp.float32), device)
        loss_value, grad_params, grad_head = loss_and_grad(
            params,
            X_batch,
            y_batch,
            meta,
            head,
            microbatch_size=None,
        )
        params, head, state = _adam_step(params, head, state, grad_params, grad_head)
        _block_until_ready((params, head, loss_value))
        losses.append(float(jax.device_get(loss_value)))
        if step == 1 or step % int(eval_every) == 0 or step == int(max_steps):
            val_scores = bmera_predict(
                params,
                head,
                dataset["val"]["X"],
                meta,
                batch_size=batch_size,
                device=device,
            )
            val_metrics = evaluate_snapshot_scores(val_scores, dataset["val"])
            selection_key = (
                val_metrics["state_balanced_accuracy"],
                val_metrics["state_auroc"],
            )
            if best is None or selection_key > best["selection_key"]:
                best = {
                    "selection_key": selection_key,
                    "step": int(step),
                    "params": np.asarray(jax.device_get(params)),
                    "head": np.asarray(jax.device_get(head)),
                }
                stale_evaluations = 0
            else:
                stale_evaluations += 1
                if stale_evaluations >= int(patience):
                    break
    fit_seconds = time.perf_counter() - fit_start
    params = jax.device_put(jnp.asarray(best["params"]), device)
    head = jax.device_put(jnp.asarray(best["head"]), device)
    split_metrics = {}
    for split_name in ("val", "iid_test", "critical_test"):
        split = dataset[split_name]
        predict_start = time.perf_counter()
        scores = bmera_predict(
            params,
            head,
            split["X"],
            meta,
            batch_size=batch_size,
            device=device,
        )
        inference_seconds = time.perf_counter() - predict_start
        metrics = evaluate_snapshot_scores(scores, split)
        split_metrics[split_name] = {
            **{key: value for key, value in metrics.items() if key != "state_predictions"},
            "inference_seconds": float(inference_seconds),
            "snapshot_scores": scores,
            "state_predictions": metrics["state_predictions"],
        }
    return {
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "init_scale": float(init_scale),
        "best_step": int(best["step"]),
        "fit_seconds": float(fit_seconds),
        "parameter_count": bmera_parameter_count(n_qubits, block_type),
        "allocated_parameter_count": bmera_parameter_count(n_qubits, block_type, allocated=True),
        "device": str(device),
        "used_cpu_fallback": bool(used_cpu_fallback),
        "losses": np.asarray(losses, dtype=np.float32),
        "params": np.asarray(jax.device_get(params)),
        "head": np.asarray(jax.device_get(head)),
        "splits": split_metrics,
    }


def tune_bmera_snapshot_classifier(
    dataset: Mapping[str, Any],
    *,
    learning_rates: Sequence[float] = (0.003, 0.01),
    init_scales: Sequence[float] = (0.1, 0.4),
    final_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    **train_kwargs,
) -> Dict[str, Any]:
    candidates = []
    for candidate_index, (learning_rate, init_scale) in enumerate(
        (pair for lr in learning_rates for pair in ((lr, scale) for scale in init_scales))
    ):
        run = train_bmera_snapshot_once(
            dataset,
            seed=91001 + candidate_index,
            learning_rate=learning_rate,
            init_scale=init_scale,
            **train_kwargs,
        )
        validation = run["splits"]["val"]
        candidates.append(
            {
                "learning_rate": float(learning_rate),
                "init_scale": float(init_scale),
                "selection_key": (
                    validation["state_balanced_accuracy"],
                    validation["state_auroc"],
                ),
            }
        )
    selected = max(candidates, key=lambda row: row["selection_key"])
    runs = [
        train_bmera_snapshot_once(
            dataset,
            seed=int(seed),
            learning_rate=selected["learning_rate"],
            init_scale=selected["init_scale"],
            **train_kwargs,
        )
        for seed in final_seeds
    ]
    return {
        "model": "bmera",
        "selected_config": {
            "learning_rate": selected["learning_rate"],
            "init_scale": selected["init_scale"],
        },
        "validation_selection_key": selected["selection_key"],
        "runs": runs,
    }


def summarize_model_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        "model": result["model"],
        "selected_config": result["selected_config"],
        "n_seeds": len(result["runs"]),
        "parameter_count_mean": float(np.mean([run["parameter_count"] for run in result["runs"]])),
        "fit_seconds_mean": float(np.mean([run["fit_seconds"] for run in result["runs"]])),
    }
    allocated_counts = [
        run["allocated_parameter_count"]
        for run in result["runs"]
        if "allocated_parameter_count" in run
    ]
    if allocated_counts:
        row["allocated_parameter_count_mean"] = float(np.mean(allocated_counts))
    for split_name in ("val", "iid_test", "critical_test"):
        for metric in (
            "snapshot_balanced_accuracy",
            "snapshot_auroc",
            "state_balanced_accuracy",
            "state_auroc",
        ):
            values = [run["splits"][split_name][metric] for run in result["runs"]]
            row[f"{split_name}_{metric}_mean"] = float(np.mean(values))
            row[f"{split_name}_{metric}_std"] = float(np.std(values))
    return row


def paired_hierarchical_bootstrap(
    bmera_result: Mapping[str, Any],
    classical_result: Mapping[str, Any],
    *,
    split_name: str = "critical_test",
    n_bootstrap: int = 10000,
    seed: int = 20260612,
) -> Dict[str, float]:
    bmera_runs = bmera_result["runs"]
    classical_runs = classical_result["runs"]
    n_seeds = min(len(bmera_runs), len(classical_runs))
    if not n_seeds:
        raise ValueError("Bootstrap requires non-empty model runs.")
    reference = bmera_runs[0]["splits"][split_name]["state_predictions"]
    state_ids = np.asarray(reference["state_id"])
    labels = np.asarray(reference["y"])
    bmera_scores = np.stack(
        [
            np.asarray(run["splits"][split_name]["state_predictions"]["score"])
            for run in bmera_runs[:n_seeds]
        ]
    )
    classical_scores = np.stack(
        [
            np.asarray(run["splits"][split_name]["state_predictions"]["score"])
            for run in classical_runs[:n_seeds]
        ]
    )
    for run in list(bmera_runs[:n_seeds]) + list(classical_runs[:n_seeds]):
        observed = np.asarray(run["splits"][split_name]["state_predictions"]["state_id"])
        if not np.array_equal(observed, state_ids):
            raise ValueError("Paired bootstrap requires identical ordered state IDs.")
    rng = np.random.default_rng(int(seed))
    differences = np.empty(int(n_bootstrap), dtype=np.float64)
    class_indices = [np.flatnonzero(labels == label) for label in (-1.0, 1.0)]
    for sample_index in range(int(n_bootstrap)):
        sampled_seeds = rng.choice(n_seeds, size=n_seeds, replace=True)
        sampled_states = np.concatenate(
            [rng.choice(indices, size=indices.size, replace=True) for indices in class_indices]
        )
        bmera_values = [
            balanced_accuracy(labels[sampled_states], bmera_scores[seed_index, sampled_states])
            for seed_index in sampled_seeds
        ]
        classical_values = [
            balanced_accuracy(labels[sampled_states], classical_scores[seed_index, sampled_states])
            for seed_index in sampled_seeds
        ]
        differences[sample_index] = np.mean(bmera_values) - np.mean(classical_values)
    observed = (
        np.mean(
            [
                run["splits"][split_name]["state_balanced_accuracy"]
                for run in bmera_runs[:n_seeds]
            ]
        )
        - np.mean(
            [
                run["splits"][split_name]["state_balanced_accuracy"]
                for run in classical_runs[:n_seeds]
            ]
        )
    )
    lower, upper = np.quantile(differences, (0.025, 0.975))
    return {
        "difference": float(observed),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_bootstrap": int(n_bootstrap),
        "supported_advantage": bool(lower > 0.0),
    }


def rank_dataset_advantages(
    benchmark_results: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int = 10000,
) -> list[Dict[str, Any]]:
    rows = []
    for dataset_result in benchmark_results:
        bmera_result = dataset_result["models"]["bmera"]
        classical = [
            result
            for name, result in dataset_result["models"].items()
            if name != "bmera"
        ]
        best_classical = max(
            classical,
            key=lambda result: summarize_model_result(result)[
                "val_state_balanced_accuracy_mean"
            ],
        )
        bootstrap = paired_hierarchical_bootstrap(
            bmera_result,
            best_classical,
            n_bootstrap=n_bootstrap,
        )
        rows.append(
            {
                "dataset": dataset_result["dataset"],
                "n_qubits": int(dataset_result["n_qubits"]),
                "best_classical": best_classical["model"],
                **bootstrap,
            }
        )
    return sorted(rows, key=lambda row: row["difference"], reverse=True)


def save_benchmark_results(
    benchmark_results: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
    output_dir: str | Path = "quantum_snapshot_runs/results",
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    summary = {"ranking": list(ranking), "datasets": []}
    for dataset_index, dataset_result in enumerate(benchmark_results):
        dataset_summary = {
            "dataset": dataset_result["dataset"],
            "n_qubits": int(dataset_result["n_qubits"]),
            "models": {},
        }
        for model_name, result in dataset_result["models"].items():
            dataset_summary["models"][model_name] = summarize_model_result(result)
            for run_index, run in enumerate(result["runs"]):
                for split_name in ("val", "iid_test", "critical_test"):
                    prediction = run["splits"][split_name]["state_predictions"]
                    prefix = f"d{dataset_index}_{model_name}_r{run_index}_{split_name}"
                    for key in ("state_id", "score", "y", "parameter"):
                        arrays[f"{prefix}_{key}"] = np.asarray(prediction[key])
        summary["datasets"].append(dataset_summary)
    summary_path = output_dir / "snapshot_benchmark_summary.json"
    predictions_path = output_dir / "snapshot_benchmark_state_predictions.npz"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(predictions_path, **arrays)
    return {"summary": str(summary_path), "predictions": str(predictions_path)}


def analytic_tfim_ridge_leakage_control(
    train_X: np.ndarray,
    train_y: np.ndarray,
    splits: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    regularization: float = 1e-3,
) -> Dict[str, float]:
    X = np.c_[np.asarray(train_X, dtype=np.float64), np.ones(len(train_y))]
    eye = np.eye(X.shape[1], dtype=np.float64)
    weights = np.linalg.solve(X.T @ X + float(regularization) * eye, X.T @ train_y)
    results = {}
    for split_name, (split_X, split_y) in splits.items():
        scores = np.c_[split_X, np.ones(len(split_y))] @ weights
        results[split_name] = balanced_accuracy(split_y, scores)
    return results


def _parameter_metadata(n_qubits: int, shape: Tuple[int, int]):
    rows = []
    for row_index in range(int(shape[0])):
        for component in range(int(shape[1])):
            rows.append(
                {
                    "row_index": row_index,
                    "layer": row_index // int(n_qubits),
                    "color": "red" if row_index % 2 == 0 else "green",
                    "block_group": (row_index % int(n_qubits)) // 2,
                    "param_component": component,
                }
            )
    return rows


def gradient_batch_statistics(
    params,
    head,
    meta: Mapping[str, Any],
    eval_batches: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    dataset_name: str,
    step: int,
    device,
    eps: float = 1e-12,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any], np.ndarray]:
    jax, jnp, _, _, loss_and_grad = _bmera_dependencies()
    gradients = []
    losses = []
    for X_eval, y_eval in eval_batches:
        X_batch = jax.device_put(jnp.asarray(X_eval, dtype=jnp.float32), device)
        y_batch = jax.device_put(jnp.asarray(y_eval, dtype=jnp.float32), device)
        loss_value, grad_params, grad_head = loss_and_grad(
            params,
            X_batch,
            y_batch,
            meta,
            head,
            microbatch_size=None,
        )
        _block_until_ready((loss_value, grad_params, grad_head))
        losses.append(float(jax.device_get(loss_value)))
        gradients.append(np.asarray(jax.device_get(grad_params), dtype=np.float64))
    gradients = np.stack(gradients)
    mean_grad = np.mean(gradients, axis=0)
    var_grad = np.var(gradients, axis=0)
    std_grad = np.std(gradients, axis=0)
    mean_abs = np.mean(np.abs(gradients), axis=0)
    rms_grad = np.sqrt(np.mean(gradients**2, axis=0))
    parameter_snr = np.abs(mean_grad) / (std_grad + float(eps))
    active_indices = list(active_block_indices(meta))
    active_rows = set(active_indices)
    active_gradients = gradients[:, active_indices, :]
    flat = active_gradients.reshape(gradients.shape[0], -1)
    flat_mean = np.mean(flat, axis=0)
    centered_noise = np.sqrt(np.mean(np.sum((flat - flat_mean) ** 2, axis=1)))
    vector_snr = np.linalg.norm(flat_mean) / (centered_noise + float(eps))
    gradient_norms = np.linalg.norm(flat, axis=1)
    n_parameters = flat.shape[1]
    n_qubits = int(meta["n_qubits"])

    parameter_rows = []
    for item in _parameter_metadata(n_qubits, gradients.shape[1:]):
        index = (item["row_index"], item["param_component"])
        parameter_rows.append(
            {
                "dataset": dataset_name,
                "n_qubits": n_qubits,
                "step": int(step),
                **item,
                "active": item["row_index"] in active_rows,
                "mean_grad": float(mean_grad[index]),
                "var_grad": float(var_grad[index]),
                "std_grad": float(std_grad[index]),
                "mean_abs_grad": float(mean_abs[index]),
                "rms_grad": float(rms_grad[index]),
                "snr": float(parameter_snr[index]),
            }
        )

    group_rows = []
    for layer in range(int(meta["depth"])):
        for color in ("red", "green"):
            selected = [
                row
                for row in parameter_rows
                if row["layer"] == layer and row["color"] == color and row["active"]
            ]
            if not selected:
                continue
            coordinates = [(row["row_index"], row["param_component"]) for row in selected]
            group_trials = np.stack(
                [gradients[:, row, component] for row, component in coordinates],
                axis=1,
            )
            group_rows.append(
                {
                    "dataset": dataset_name,
                    "n_qubits": n_qubits,
                    "step": int(step),
                    "layer": layer,
                    "color": color,
                    "n_parameters": len(selected),
                    "normalized_gradient_rms": float(np.sqrt(np.mean(group_trials**2))),
                    "mean_parameter_variance": float(
                        np.mean([row["var_grad"] for row in selected])
                    ),
                    "median_parameter_snr": float(
                        np.median([row["snr"] for row in selected])
                    ),
                    "mean_absolute_gradient": float(
                        np.mean([row["mean_abs_grad"] for row in selected])
                    ),
                }
            )
    heatmap = mean_abs.reshape(int(meta["depth"]), n_qubits * int(meta["n_per_block"]))
    global_row = {
        "dataset": dataset_name,
        "n_qubits": n_qubits,
        "step": int(step),
        "eval_loss_mean": float(np.mean(losses)),
        "eval_loss_std": float(np.std(losses)),
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "gradient_norm_std": float(np.std(gradient_norms)),
        "normalized_gradient_rms": float(np.sqrt(np.mean(active_gradients**2))),
        "mean_parameter_variance": float(np.mean(var_grad[active_indices])),
        "median_parameter_snr": float(np.median(parameter_snr[active_indices])),
        "vector_snr": float(vector_snr),
        "mean_absolute_gradient": float(np.mean(mean_abs[active_indices])),
        "n_parameters": int(n_parameters),
    }
    finite = np.asarray(
        [
            global_row["normalized_gradient_rms"],
            global_row["mean_parameter_variance"],
            global_row["median_parameter_snr"],
            global_row["vector_snr"],
            global_row["mean_absolute_gradient"],
        ]
    )
    if not np.all(np.isfinite(finite)):
        raise RuntimeError(f"Non-finite gradient diagnostics for {dataset_name}.")
    return parameter_rows, group_rows, global_row, heatmap


def run_snapshot_path_diagnostics(
    dataset: Mapping[str, Any],
    *,
    snapshot_steps: Sequence[int] = (0, 1, 2, 3, 5, 8, 13, 21, 34, 50, 75, 100),
    eval_batch_size: int = 128,
    eval_batches: int = 8,
    train_batch_size: int = 128,
    learning_rate: float = 0.01,
    init_scale: float = 0.1,
    seed: int = 0,
    prefer_gpu: bool = True,
) -> Dict[str, Any]:
    jax, jnp, _, _, loss_and_grad = _bmera_dependencies()
    snapshot_steps = tuple(int(step) for step in snapshot_steps)
    if snapshot_steps != tuple(sorted(snapshot_steps)):
        raise ValueError("snapshot_steps must be sorted.")
    n_qubits = int(dataset["metadata"]["n_qubits"])
    dataset_name = str(dataset["metadata"]["model"])
    device, used_cpu_fallback = select_jax_device(prefer_gpu=prefer_gpu)
    meta, params, head = _init_bmera(n_qubits, seed, init_scale, block_type=1)
    params = jax.device_put(params, device)
    head = jax.device_put(head, device)
    state = _init_adam(params, head, learning_rate)
    rng = np.random.default_rng(int(seed) + 80021)
    val = dataset["val"]
    fixed_batches = []
    for _ in range(int(eval_batches)):
        indices = rng.choice(
            val["X"].shape[0],
            size=int(eval_batch_size),
            replace=val["X"].shape[0] < int(eval_batch_size),
        )
        fixed_batches.append((val["X"][indices], val["y"][indices]))
    parameter_rows = []
    group_rows = []
    global_rows = []
    heatmaps = {}
    train = dataset["train"]
    latest_train_loss = float("nan")

    def record(step):
        params_rows, groups, global_row, heatmap = gradient_batch_statistics(
            params,
            head,
            meta,
            fixed_batches,
            dataset_name=dataset_name,
            step=step,
            device=device,
        )
        global_row.update(
            {
                "train_loss": latest_train_loss,
                "device": str(device),
                "used_cpu_fallback": bool(used_cpu_fallback),
                "eval_batches": int(eval_batches),
                "eval_batch_size": int(eval_batch_size),
            }
        )
        parameter_rows.extend(params_rows)
        group_rows.extend(groups)
        global_rows.append(global_row)
        heatmaps[(dataset_name, n_qubits, int(step))] = heatmap

    if 0 in snapshot_steps:
        record(0)
    for step in range(1, max(snapshot_steps) + 1):
        indices = rng.choice(
            train["X"].shape[0],
            size=int(train_batch_size),
            replace=train["X"].shape[0] < int(train_batch_size),
        )
        X_batch = jax.device_put(jnp.asarray(train["X"][indices], dtype=jnp.float32), device)
        y_batch = jax.device_put(jnp.asarray(train["y"][indices], dtype=jnp.float32), device)
        loss_value, grad_params, grad_head = loss_and_grad(
            params,
            X_batch,
            y_batch,
            meta,
            head,
            microbatch_size=None,
        )
        params, head, state = _adam_step(params, head, state, grad_params, grad_head)
        _block_until_ready((params, head, loss_value))
        latest_train_loss = float(jax.device_get(loss_value))
        if step in snapshot_steps:
            record(step)
    return {
        "parameter_rows": parameter_rows,
        "group_rows": group_rows,
        "global_rows": global_rows,
        "heatmaps": heatmaps,
        "device": device,
    }
