"""Analytic TFIM descriptors and the leakage-control workflow."""

import jax
import jax.numpy as jnp
import numpy as np
from bmera import build_bmera_meta
from .synthetic import (
    block_tree_until_ready_local,
    evaluate_bmera_classifier,
    init_adam_state,
    synthetic_target_device,
    train_synthetic_step,
)

# Transverse-field Ising model (TFIM) phase-classification pipeline.
# The generator uses the exact free-fermion Bogoliubov angles, so no external
# many-body or sklearn dependency is required.

import json
from pathlib import Path


TFIM_QUBITS = (8, 16, 32, 64)
TFIM_TYPE = 1
TFIM_OPTIMIZER = 'adam'
TFIM_REPEATS = 3
TFIM_STEPS = 300
TFIM_LR = 1e-2
TFIM_LOG_EVERY = 50
TFIM_DTYPE = jnp.float32
TFIM_N_TRAIN = 4096
TFIM_N_VAL = 1024
TFIM_N_TEST = 2048
TFIM_N_CRITICAL_TEST = 2048
TFIM_BATCH_SIZE_BY_QUBITS = {
    8: 512,
    16: 256,
    32: 128,
    64: 64,
}
TFIM_TRAIN_G_RANGES = {
    -1.0: (0.25, 0.85),
    1.0: (1.15, 1.75),
}
TFIM_CRITICAL_G_RANGES = {
    -1.0: (0.90, 0.99),
    1.0: (1.01, 1.10),
}
TFIM_ANGLE_NOISE_STD = 0.02
TFIM_BASE_SEED = 314159
TFIM_SAVE_WEIGHTS = True
TFIM_SAVE_DIR = Path('tfim_bmera_runs')


def tfim_bogoliubov_angles(g_values, n_modes):
    # Exact ground-state Bogoliubov angles for clean 1D TFIM samples.
    g_values = np.asarray(g_values, dtype=np.float64).reshape(-1, 1)
    momenta = np.pi * (np.arange(int(n_modes), dtype=np.float64) + 0.5) / int(n_modes)
    angles = 0.5 * np.arctan2(
        np.sin(momenta)[None, :],
        g_values - np.cos(momenta)[None, :],
    )
    return angles.astype(np.float32), momenta.astype(np.float32)


def _sample_balanced_tfim_split(n_samples, n_qubits, rng, phase_ranges, noise_std):
    n_ferro = int(n_samples) // 2
    n_para = int(n_samples) - n_ferro
    g_ferro = rng.uniform(*phase_ranges[-1.0], size=n_ferro)
    g_para = rng.uniform(*phase_ranges[1.0], size=n_para)
    g_values = np.concatenate([g_ferro, g_para]).astype(np.float32)
    labels = np.concatenate([
        -np.ones(n_ferro, dtype=np.float32),
        np.ones(n_para, dtype=np.float32),
    ])
    features, momenta = tfim_bogoliubov_angles(g_values, n_qubits)
    if noise_std:
        features = features + rng.normal(0.0, noise_std, size=features.shape).astype(np.float32)
    features = np.clip(features, 0.0, 0.5 * np.pi).astype(np.float32)
    permutation = rng.permutation(int(n_samples))
    return {
        'X': features[permutation],
        'y': labels[permutation],
        'g': g_values[permutation],
        'momenta': momenta,
    }


def make_tfim_phase_dataset(
    n_qubits,
    n_train=TFIM_N_TRAIN,
    n_val=TFIM_N_VAL,
    n_test=TFIM_N_TEST,
    n_critical_test=TFIM_N_CRITICAL_TEST,
    seed=TFIM_BASE_SEED,
    noise_std=TFIM_ANGLE_NOISE_STD,
):
    rng = np.random.default_rng(int(seed) + 101 * int(n_qubits))
    return {
        'name': 'tfim_phase',
        'n_qubits': int(n_qubits),
        'train': _sample_balanced_tfim_split(
            n_train, n_qubits, rng, TFIM_TRAIN_G_RANGES, noise_std
        ),
        'val': _sample_balanced_tfim_split(
            n_val, n_qubits, rng, TFIM_TRAIN_G_RANGES, noise_std
        ),
        'test': _sample_balanced_tfim_split(
            n_test, n_qubits, rng, TFIM_TRAIN_G_RANGES, noise_std
        ),
        'critical_test': _sample_balanced_tfim_split(
            n_critical_test, n_qubits, rng, TFIM_CRITICAL_G_RANGES, noise_std
        ),
    }


def assert_tfim_dataset_ok(dataset):
    n_qubits = int(dataset['n_qubits'])
    for split_name in ('train', 'val', 'test', 'critical_test'):
        split = dataset[split_name]
        X = np.asarray(split['X'])
        y = np.asarray(split['y'])
        g = np.asarray(split['g'])
        assert X.shape == (y.size, n_qubits), (split_name, X.shape, y.shape)
        assert g.shape == y.shape, (split_name, g.shape, y.shape)
        assert set(np.unique(y).tolist()) == {-1.0, 1.0}, (split_name, np.unique(y))
        assert np.all(np.isfinite(X)), split_name
        assert np.all((X >= 0.0) & (X <= 0.5 * np.pi)), split_name
        assert np.all((g < 1.0) == (y < 0.0)), split_name


def initialize_tfim_bmera_params(n_qubits, repeat, type_=TFIM_TYPE):
    meta = build_bmera_meta(int(n_qubits), int(type_))
    rng = np.random.default_rng(TFIM_BASE_SEED + 1009 * int(n_qubits) + 9176 * int(repeat))
    param_shape = (int(meta['depth']) * int(n_qubits), int(meta['n_per_block']))
    params = jnp.asarray(rng.uniform(-0.4, 0.4, size=param_shape), dtype=TFIM_DTYPE)
    head = jnp.asarray(rng.uniform(-0.4, 0.4, size=(int(n_qubits),)), dtype=TFIM_DTYPE)
    return meta, params, head


def evaluate_tfim_split(params, head, split, meta, batch_size, device=None):
    result = evaluate_bmera_classifier(
        params,
        head,
        split['X'],
        split['y'],
        meta,
        batch_size=batch_size,
        device=device,
    )
    result['g'] = np.asarray(split['g'])
    return result


def save_tfim_training_artifacts(result, save_dir=TFIM_SAVE_DIR):
    if result.get('status') != 'ok':
        return None
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"tfim_n{int(result['n_qubits'])}_{result['optimizer']}_"
        f"repeat{int(result['repeat'])}_steps{int(result['steps'])}"
    )
    weights_path = save_dir / f'{stem}.npz'
    metadata_path = save_dir / f'{stem}.json'
    np.savez_compressed(
        weights_path,
        params=np.asarray(result['params']),
        head_weights=np.asarray(result['head_weights']),
        losses=np.asarray(result['losses'], dtype=np.float32),
    )
    metadata = {
        key: (value.item() if isinstance(value, (np.integer, np.floating)) else value)
        for key, value in result.items()
        if key not in {'params', 'head_weights', 'losses'}
    }
    metadata['weights_path'] = str(weights_path)
    metadata['metadata_path'] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return {'weights_path': weights_path, 'metadata_path': metadata_path}


def train_tfim_bmera_once(
    n_qubits,
    repeat=0,
    steps=TFIM_STEPS,
    lr=TFIM_LR,
    batch_size=None,
    log_every=TFIM_LOG_EVERY,
    save_weights=TFIM_SAVE_WEIGHTS,
):
    n_qubits = int(n_qubits)
    if batch_size is None:
        batch_size = int(TFIM_BATCH_SIZE_BY_QUBITS[n_qubits])
    dataset = make_tfim_phase_dataset(n_qubits)
    assert_tfim_dataset_ok(dataset)
    meta, params, head = initialize_tfim_bmera_params(n_qubits, repeat)
    device = synthetic_target_device()
    if device is not None:
        params = jax.device_put(params, device)
        head = jax.device_put(head, device)
    state = init_adam_state(params, head, lr)

    X_train = dataset['train']['X']
    y_train = dataset['train']['y']
    rng = np.random.default_rng(TFIM_BASE_SEED + 50021 * n_qubits + 1237 * int(repeat))
    permutation = rng.permutation(X_train.shape[0])
    cursor = 0
    losses = []

    for step_idx in range(int(steps)):
        if cursor + batch_size > X_train.shape[0]:
            permutation = rng.permutation(X_train.shape[0])
            cursor = 0
        batch_idx = permutation[cursor:cursor + batch_size]
        cursor += batch_size
        X_batch = jnp.asarray(X_train[batch_idx], dtype=TFIM_DTYPE)
        y_batch = jnp.asarray(y_train[batch_idx], dtype=TFIM_DTYPE)
        if device is not None:
            X_batch = jax.device_put(X_batch, device)
            y_batch = jax.device_put(y_batch, device)
        params, head, state, lval = train_synthetic_step(
            params,
            head,
            state,
            X_batch,
            y_batch,
            meta,
            optimizer=TFIM_OPTIMIZER,
            microbatch_size=None,
        )
        block_tree_until_ready_local((params, head, lval))
        current_loss = float(jax.device_get(lval))
        losses.append(current_loss)
        if log_every and ((step_idx + 1) == 1 or (step_idx + 1) % int(log_every) == 0):
            print(
                f'TFIM train | n={n_qubits:2d} | repeat={repeat} | '
                f'step={step_idx + 1:4d}/{int(steps)} | loss={current_loss:.6f}'
            )

    val = evaluate_tfim_split(params, head, dataset['val'], meta, batch_size, device=device)
    test = evaluate_tfim_split(params, head, dataset['test'], meta, batch_size, device=device)
    critical = evaluate_tfim_split(
        params, head, dataset['critical_test'], meta, batch_size, device=device
    )
    result = {
        'status': 'ok',
        'dataset': 'tfim_phase',
        'n_qubits': n_qubits,
        'bmera_type': int(TFIM_TYPE),
        'optimizer': TFIM_OPTIMIZER,
        'repeat': int(repeat),
        'steps': int(steps),
        'learning_rate': float(lr),
        'train_batch_size': int(batch_size),
        'angle_noise_std': float(TFIM_ANGLE_NOISE_STD),
        'val_accuracy': float(val['accuracy']),
        'val_mse': float(val['mse']),
        'test_accuracy': float(test['accuracy']),
        'test_mse': float(test['mse']),
        'critical_test_accuracy': float(critical['accuracy']),
        'critical_test_mse': float(critical['mse']),
        'final_loss': float(losses[-1]) if losses else np.nan,
        'device': str(device) if device is not None else str(jax.devices('cpu')[0]),
        'params': params,
        'head_weights': head,
        'losses': np.asarray(losses, dtype=np.float32),
    }
    if save_weights:
        paths = save_tfim_training_artifacts(result)
        if paths:
            print(f"TFIM saved | weights={paths['weights_path']} | metadata={paths['metadata_path']}")
    return result


def run_tfim_bmera_sweep(
    qubits=TFIM_QUBITS,
    repeats=TFIM_REPEATS,
    steps=TFIM_STEPS,
    save_weights=TFIM_SAVE_WEIGHTS,
):
    results = []
    for n_qubits in qubits:
        for repeat in range(int(repeats)):
            print('=' * 88)
            result = train_tfim_bmera_once(
                n_qubits=n_qubits,
                repeat=repeat,
                steps=steps,
                save_weights=save_weights,
            )
            print(
                f"TFIM result | n={n_qubits} | repeat={repeat} | "
                f"val={result['val_accuracy']:.3f} | test={result['test_accuracy']:.3f} | "
                f"near-critical={result['critical_test_accuracy']:.3f}"
            )
            results.append(result)
    return results


def print_tfim_results_table(results):
    if not results:
        print('No TFIM training results.')
        return
    print('qubits | repeat | val acc | test acc | near-critical acc | final loss')
    print('-' * 72)
    for row in sorted(results, key=lambda r: (r['n_qubits'], r['repeat'])):
        print(
            f"{row['n_qubits']:>6d} | {row['repeat']:>6d} | "
            f"{row['val_accuracy']:.3f} | {row['test_accuracy']:.3f} | "
            f"{row['critical_test_accuracy']:.3f} | {row['final_loss']:.4f}"
        )
