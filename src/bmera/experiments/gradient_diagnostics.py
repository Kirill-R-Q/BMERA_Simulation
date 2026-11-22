"""Gradient statistics across sizes and along training trajectories."""

import jax
import jax.numpy as jnp
import numpy as np
from bmera import build_bmera_meta, loss_and_grad_autodiff_microbatched
from bmera.topology import active_block_indices

from .synthetic import (
    SYNTHETIC_BASE_SEEDS,
    SYNTHETIC_BATCH_SIZE_BY_QUBITS,
    SYNTHETIC_DTYPE,
    SYNTHETIC_LR_BY_OPTIMIZER,
    SYNTHETIC_TYPE,
    assert_synthetic_dataset_ok,
    block_tree_until_ready_local,
    init_adam_state,
    initialize_bmera_params,
    prepare_synthetic_dataset,
    train_synthetic_step,
)

# Gradient diagnostics configuration and helpers.
# The sweep prefers a JAX GPU device, but falls back to CPU when this environment
# does not expose a GPU backend.

GRAD_DIAG_QUBITS = (8, 16, 32, 64)
GRAD_DIAG_TYPES = (1, 2, 3)
GRAD_DIAG_TRIALS_FAST = 5
GRAD_DIAG_BATCH_FAST = 8
GRAD_DIAG_TRIALS_PAPER = 32
GRAD_DIAG_BATCH_PAPER = 32
GRAD_DIAG_PARAM_RANGE = (-0.5, 0.5)
GRAD_DIAG_ANGLE_RANGE = (-np.pi, np.pi)
GRAD_DIAG_TARGET_RANGE = (-1.0, 1.0)
GRAD_DIAG_HEAD_RANGE = (-0.5, 0.5)
GRAD_DIAG_EPS = 1e-12


def select_grad_diag_device(prefer_gpu=True):
    gpu_devices = [device for device in jax.devices() if device.platform in ('gpu', 'cuda', 'rocm')]
    if prefer_gpu and gpu_devices:
        return gpu_devices[0], False
    cpu_devices = jax.devices('cpu')
    if not cpu_devices:
        raise RuntimeError('No JAX CPU device is available for the gradient diagnostics fallback.')
    return cpu_devices[0], True


def _grad_diag_param_metadata(n_qubits, params_shape):
    rows, n_components = params_shape
    meta_rows = []
    seen = set()
    for row_index in range(int(rows)):
        layer = row_index // int(n_qubits)
        color = 'red' if row_index % 2 == 0 else 'green'
        block_group = (row_index % int(n_qubits)) // 2
        for param_component in range(int(n_components)):
            key = (int(row_index), int(param_component))
            seen.add(key)
            meta_rows.append({
                'row_index': int(row_index),
                'layer': int(layer),
                'color': color,
                'block_group': int(block_group),
                'param_component': int(param_component),
            })
    expected = int(rows) * int(n_components)
    if len(seen) != expected:
        raise RuntimeError(f'Parameter metadata coverage mismatch: {len(seen)} != {expected}')
    return meta_rows


def _grad_diag_trial(rng, n_qubits, block_type, batch_size, device, dtype=jnp.float32):
    meta = build_bmera_meta(int(n_qubits), int(block_type))
    param_shape = (int(meta['depth']) * int(n_qubits), int(meta['n_per_block']))
    params = jnp.asarray(rng.uniform(*GRAD_DIAG_PARAM_RANGE, size=param_shape), dtype=dtype)
    X = jnp.asarray(rng.uniform(*GRAD_DIAG_ANGLE_RANGE, size=(int(batch_size), int(n_qubits))), dtype=dtype)
    y = jnp.asarray(rng.uniform(*GRAD_DIAG_TARGET_RANGE, size=(int(batch_size),)), dtype=dtype)
    head = jnp.asarray(rng.uniform(*GRAD_DIAG_HEAD_RANGE, size=(int(n_qubits),)), dtype=dtype)

    params = jax.device_put(params, device)
    X = jax.device_put(X, device)
    y = jax.device_put(y, device)
    head = jax.device_put(head, device)

    loss_value, grad_params, grad_head = loss_and_grad_autodiff_microbatched(
        params, X, y, meta, head, microbatch_size=None
    )
    if hasattr(loss_value, 'block_until_ready'):
        loss_value.block_until_ready()
    if hasattr(grad_params, 'block_until_ready'):
        grad_params.block_until_ready()
    grad_np = np.asarray(jax.device_get(grad_params), dtype=np.float64)
    if grad_np.shape != param_shape:
        raise RuntimeError(f'Gradient shape mismatch for n={n_qubits}, type={block_type}: {grad_np.shape} != {param_shape}')
    return float(jax.device_get(loss_value)), grad_np, meta


def _mean_rows(rows, key):
    vals = [float(row[key]) for row in rows]
    return float(np.mean(vals)) if vals else np.nan


def _std_rows(rows, key):
    vals = [float(row[key]) for row in rows]
    return float(np.std(vals)) if vals else np.nan


def run_gradient_diagnostics(
    qubits=GRAD_DIAG_QUBITS,
    block_types=GRAD_DIAG_TYPES,
    trials=GRAD_DIAG_TRIALS_FAST,
    batch_size=GRAD_DIAG_BATCH_FAST,
    seed=2026,
    prefer_gpu=True,
    label='fast',
):
    device, used_cpu_fallback = select_grad_diag_device(prefer_gpu=prefer_gpu)
    if used_cpu_fallback:
        print(f'No JAX GPU backend available; running gradient diagnostics on CPU device: {device}')
    else:
        print(f'Running gradient diagnostics on device: {device}')

    rng = np.random.default_rng(int(seed))
    param_rows = []
    group_rows = []
    global_rows = []

    for block_type in block_types:
        for n_qubits in qubits:
            print(f'[{label}] n={n_qubits}, block_type={block_type}, trials={trials}, batch={batch_size}')
            trial_grads = []
            trial_losses = []
            meta = None
            for _ in range(int(trials)):
                loss_value, grad_np, meta = _grad_diag_trial(
                    rng=rng,
                    n_qubits=int(n_qubits),
                    block_type=int(block_type),
                    batch_size=int(batch_size),
                    device=device,
                )
                trial_losses.append(loss_value)
                trial_grads.append(grad_np)

            grads = np.stack(trial_grads, axis=0)
            active_indices = list(active_block_indices(meta))
            active_rows = set(active_indices)
            flat = grads[:, active_indices, :].reshape(int(trials), -1)
            mean_grad = np.mean(grads, axis=0)
            var_grad = np.var(grads, axis=0)
            std_grad = np.std(grads, axis=0)
            rms_grad = np.sqrt(np.mean(grads * grads, axis=0))
            mean_abs_grad = np.mean(np.abs(grads), axis=0)
            snr = np.abs(mean_grad) / (std_grad + GRAD_DIAG_EPS)
            trial_norms = np.linalg.norm(flat, axis=1)

            param_meta = _grad_diag_param_metadata(int(n_qubits), grads.shape[1:])
            for item in param_meta:
                idx = (item['row_index'], item['param_component'])
                param_rows.append({
                    'label': label,
                    'device': str(device),
                    'used_cpu_fallback': bool(used_cpu_fallback),
                    'n_qubits': int(n_qubits),
                    'block_type': int(block_type),
                    **item,
                    'active': item['row_index'] in active_rows,
                    'mean_grad': float(mean_grad[idx]),
                    'var_grad': float(var_grad[idx]),
                    'std_grad': float(std_grad[idx]),
                    'rms_grad': float(rms_grad[idx]),
                    'mean_abs_grad': float(mean_abs_grad[idx]),
                    'snr': float(snr[idx]),
                })

            for layer in range(int(meta['depth'])):
                for color in ('red', 'green'):
                    mask_rows = [
                        i for i, item in enumerate(param_meta)
                        if item['layer'] == layer and item['color'] == color and item['row_index'] in active_rows
                    ]
                    if not mask_rows:
                        continue
                    coords = [(param_meta[i]['row_index'], param_meta[i]['param_component']) for i in mask_rows]
                    group_trial = np.stack([grads[:, row, col] for row, col in coords], axis=1)
                    group_norms = np.linalg.norm(group_trial, axis=1)
                    group_param_rows = [
                        row for row in param_rows
                        if row['label'] == label
                        and row['n_qubits'] == int(n_qubits)
                        and row['block_type'] == int(block_type)
                        and row['layer'] == int(layer)
                        and row['color'] == color
                        and row['active']
                    ]
                    group_rows.append({
                        'label': label,
                        'device': str(device),
                        'used_cpu_fallback': bool(used_cpu_fallback),
                        'n_qubits': int(n_qubits),
                        'block_type': int(block_type),
                        'layer': int(layer),
                        'color': color,
                        'n_parameters': int(len(coords)),
                        'grad_norm_mean': float(np.mean(group_norms)),
                        'grad_norm_std': float(np.std(group_norms)),
                        'mean_param_var': _mean_rows(group_param_rows, 'var_grad'),
                        'mean_param_snr': _mean_rows(group_param_rows, 'snr'),
                        'mean_param_rms': _mean_rows(group_param_rows, 'rms_grad'),
                    })

            current_param_rows = [
                row for row in param_rows
                if row['label'] == label and row['n_qubits'] == int(n_qubits) and row['block_type'] == int(block_type)
                and row['active']
            ]
            finite_checks = [row['var_grad'] for row in current_param_rows] + [row['snr'] for row in current_param_rows]
            if not np.all(np.isfinite(np.asarray(finite_checks, dtype=np.float64))):
                raise RuntimeError(f'Non-finite gradient diagnostics for n={n_qubits}, type={block_type}')
            global_rows.append({
                'label': label,
                'device': str(device),
                'used_cpu_fallback': bool(used_cpu_fallback),
                'n_qubits': int(n_qubits),
                'block_type': int(block_type),
                'trials': int(trials),
                'batch_size': int(batch_size),
                'loss_mean': float(np.mean(trial_losses)),
                'loss_std': float(np.std(trial_losses)),
                'grad_norm_mean': float(np.mean(trial_norms)),
                'grad_norm_std': float(np.std(trial_norms)),
                'mean_param_var': _mean_rows(current_param_rows, 'var_grad'),
                'mean_param_snr': _mean_rows(current_param_rows, 'snr'),
                'mean_param_rms': _mean_rows(current_param_rows, 'rms_grad'),
                'mean_param_abs_grad': _mean_rows(current_param_rows, 'mean_abs_grad'),
            })

    return param_rows, group_rows, global_rows, device


def print_gradient_diag_global_table(rows):
    if not rows:
        print('No gradient diagnostic rows to summarize.')
        return
    header = 'type | qubits | norm mean | norm std | mean var | mean snr | mean rms | device'
    print(header)
    print('-' * len(header))
    for row in sorted(rows, key=lambda r: (r['block_type'], r['n_qubits'])):
        print(
            f"{row['block_type']:>4d} | {row['n_qubits']:>6d} | "
            f"{row['grad_norm_mean']:.3e} | {row['grad_norm_std']:.3e} | "
            f"{row['mean_param_var']:.3e} | {row['mean_param_snr']:.3e} | "
            f"{row['mean_param_rms']:.3e} | {row['device']}"
        )


def print_gradient_diag_layer_table(rows, n_qubits=64, block_type=1):
    selected = [
        row for row in rows
        if row['n_qubits'] == int(n_qubits) and row['block_type'] == int(block_type)
    ]
    if not selected:
        print(f'No layer/color rows for n={n_qubits}, block_type={block_type}.')
        return
    print(f'Layer/color summary for n={n_qubits}, block_type={block_type}')
    print('layer | color | norm mean | norm std | mean var | mean snr | mean rms')
    print('-' * 78)
    for row in sorted(selected, key=lambda r: (r['layer'], r['color'])):
        print(
            f"{row['layer']:>5d} | {row['color']:>5s} | "
            f"{row['grad_norm_mean']:.3e} | {row['grad_norm_std']:.3e} | "
            f"{row['mean_param_var']:.3e} | {row['mean_param_snr']:.3e} | {row['mean_param_rms']:.3e}"
        )


# Moons early-training gradient diagnostics.
# Set RUN_MOONS_PATH_SMOKE = True for a quick validation run before the full default.

MOONS_PATH_QUBITS = (8, 16)
MOONS_PATH_OPTIMIZER = 'adam'
MOONS_PATH_TYPE = SYNTHETIC_TYPE
MOONS_PATH_REPEAT = 0
MOONS_PATH_LR = SYNTHETIC_LR_BY_OPTIMIZER['adam']
MOONS_PATH_SNAPSHOT_STEPS = (0, 1, 2, 3, 5, 8, 13, 21, 34, 50, 75, 100, 120)
MOONS_PATH_EVAL_BATCH_SIZE = 256
MOONS_PATH_EVAL_BATCHES = 8
MOONS_PATH_TRAIN_SEED_OFFSET = 70001
MOONS_PATH_EVAL_SEED_OFFSET = 80021
MOONS_PATH_EPS = GRAD_DIAG_EPS


def _moons_path_config(smoke=False):
    if smoke:
        return {
            'qubits': (8,),
            'snapshot_steps': (0, 1),
            'eval_batch_size': 64,
            'eval_batches': 2,
        }
    return {
        'qubits': MOONS_PATH_QUBITS,
        'snapshot_steps': MOONS_PATH_SNAPSHOT_STEPS,
        'eval_batch_size': MOONS_PATH_EVAL_BATCH_SIZE,
        'eval_batches': MOONS_PATH_EVAL_BATCHES,
    }


def _fixed_eval_batches(X_train, y_train, n_qubits, batch_size, n_batches):
    rng = np.random.default_rng(SYNTHETIC_BASE_SEEDS['moons'] + MOONS_PATH_EVAL_SEED_OFFSET + int(n_qubits))
    batches = []
    for _ in range(int(n_batches)):
        idx = rng.choice(X_train.shape[0], size=int(batch_size), replace=False)
        batches.append((X_train[idx], y_train[idx]))
    return batches


def _moons_path_heatmap(mean_abs_grad, n_qubits, meta):
    depth = int(meta['depth'])
    n_per_block = int(meta['n_per_block'])
    heatmap = np.zeros((depth, int(n_qubits) * n_per_block), dtype=np.float64)
    for row_index in range(mean_abs_grad.shape[0]):
        layer = row_index // int(n_qubits)
        row_within_layer = row_index % int(n_qubits)
        start = row_within_layer * n_per_block
        heatmap[layer, start:start + n_per_block] = mean_abs_grad[row_index]
    expected_shape = (depth, int(n_qubits) * n_per_block)
    if heatmap.shape != expected_shape:
        raise RuntimeError(f'Unexpected heatmap shape: {heatmap.shape} != {expected_shape}')
    return heatmap


def _moons_eval_gradient_stats(params, head, meta, eval_batches, n_qubits, step, device):
    grads = []
    losses = []
    for X_eval, y_eval in eval_batches:
        X_batch = jnp.asarray(X_eval, dtype=SYNTHETIC_DTYPE)
        y_batch = jnp.asarray(y_eval, dtype=SYNTHETIC_DTYPE)
        X_batch = jax.device_put(X_batch, device)
        y_batch = jax.device_put(y_batch, device)
        lval, grad_params, grad_head = loss_and_grad_autodiff_microbatched(
            params,
            X_batch,
            y_batch,
            meta,
            head,
            microbatch_size=None,
        )
        block_tree_until_ready_local((lval, grad_params, grad_head))
        losses.append(float(jax.device_get(lval)))
        grads.append(np.asarray(jax.device_get(grad_params), dtype=np.float64))

    grads = np.stack(grads, axis=0)
    if grads.shape[1:] != tuple(params.shape):
        raise RuntimeError(f'Gradient shape mismatch at step={step}: {grads.shape[1:]} != {tuple(params.shape)}')

    active_indices = list(active_block_indices(meta))
    active_rows = set(active_indices)
    flat = grads[:, active_indices, :].reshape(grads.shape[0], -1)
    mean_grad = np.mean(grads, axis=0)
    var_grad = np.var(grads, axis=0)
    std_grad = np.std(grads, axis=0)
    mean_abs_grad = np.mean(np.abs(grads), axis=0)
    rms_grad = np.sqrt(np.mean(grads * grads, axis=0))
    snr = np.abs(mean_grad) / (std_grad + MOONS_PATH_EPS)
    grad_norms = np.linalg.norm(flat, axis=1)

    param_rows = []
    param_meta = _grad_diag_param_metadata(int(n_qubits), grads.shape[1:])
    if len(param_meta) != int(np.prod(grads.shape[1:])):
        raise RuntimeError('Parameter metadata does not cover every gradient parameter.')
    for item in param_meta:
        idx = (item['row_index'], item['param_component'])
        param_rows.append({
            'dataset': 'moons',
            'optimizer': MOONS_PATH_OPTIMIZER,
            'n_qubits': int(n_qubits),
            'block_type': int(MOONS_PATH_TYPE),
            'step': int(step),
            **item,
            'active': item['row_index'] in active_rows,
            'mean_grad': float(mean_grad[idx]),
            'var_grad': float(var_grad[idx]),
            'std_grad': float(std_grad[idx]),
            'mean_abs_grad': float(mean_abs_grad[idx]),
            'rms_grad': float(rms_grad[idx]),
            'snr': float(snr[idx]),
        })

    group_rows = []
    for layer in range(int(meta['depth'])):
        for color in ('red', 'green'):
            rows = [row for row in param_rows if row['layer'] == layer and row['color'] == color and row['active']]
            if not rows:
                continue
            coords = [(row['row_index'], row['param_component']) for row in rows]
            group_trial = np.stack([grads[:, row, col] for row, col in coords], axis=1)
            group_norms = np.linalg.norm(group_trial, axis=1)
            group_rows.append({
                'dataset': 'moons',
                'optimizer': MOONS_PATH_OPTIMIZER,
                'n_qubits': int(n_qubits),
                'block_type': int(MOONS_PATH_TYPE),
                'step': int(step),
                'layer': int(layer),
                'color': color,
                'n_parameters': int(len(rows)),
                'grad_norm_mean': float(np.mean(group_norms)),
                'grad_norm_std': float(np.std(group_norms)),
                'mean_param_var': float(np.mean([row['var_grad'] for row in rows])),
                'mean_param_snr': float(np.mean([row['snr'] for row in rows])),
                'mean_param_abs_grad': float(np.mean([row['mean_abs_grad'] for row in rows])),
                'mean_param_rms': float(np.mean([row['rms_grad'] for row in rows])),
            })

    finite_values = np.asarray(
        [row['var_grad'] for row in param_rows]
        + [row['snr'] for row in param_rows]
        + [row['mean_abs_grad'] for row in param_rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(finite_values)):
        raise RuntimeError(f'Non-finite gradient diagnostic values at n={n_qubits}, step={step}')

    global_row = {
        'dataset': 'moons',
        'optimizer': MOONS_PATH_OPTIMIZER,
        'n_qubits': int(n_qubits),
        'block_type': int(MOONS_PATH_TYPE),
        'step': int(step),
        'eval_loss_mean': float(np.mean(losses)),
        'eval_loss_std': float(np.std(losses)),
        'grad_norm_mean': float(np.mean(grad_norms)),
        'grad_norm_std': float(np.std(grad_norms)),
        'mean_param_var': float(np.mean(var_grad[active_indices])),
        'mean_param_snr': float(np.mean(snr[active_indices])),
        'mean_param_abs_grad': float(np.mean(mean_abs_grad[active_indices])),
        'mean_param_rms': float(np.mean(rms_grad[active_indices])),
    }
    heatmap = _moons_path_heatmap(mean_abs_grad, n_qubits, meta)
    return param_rows, group_rows, global_row, heatmap


def run_moons_training_path_diagnostics(smoke=False):
    config = _moons_path_config(smoke=smoke)
    snapshot_steps = tuple(int(s) for s in config['snapshot_steps'])
    if tuple(sorted(snapshot_steps)) != snapshot_steps:
        raise ValueError('MOONS_PATH_SNAPSHOT_STEPS must be sorted.')

    device, used_cpu_fallback = select_grad_diag_device(prefer_gpu=True)
    if used_cpu_fallback:
        print(f'No JAX GPU backend available; running moons path diagnostics on CPU device: {device}')
    else:
        print(f'Running moons path diagnostics on device: {device}')

    param_rows = []
    group_rows = []
    global_rows = []
    heatmap_data = {}

    for n_qubits in config['qubits']:
        prepared = prepare_synthetic_dataset('moons', int(n_qubits))
        assert_synthetic_dataset_ok(prepared)
        X_train, y_train = prepared['train']
        eval_batches = _fixed_eval_batches(
            X_train,
            y_train,
            n_qubits=int(n_qubits),
            batch_size=int(config['eval_batch_size']),
            n_batches=int(config['eval_batches']),
        )
        meta, params, head = initialize_bmera_params(int(n_qubits), MOONS_PATH_REPEAT, 'moons', type_=MOONS_PATH_TYPE)
        params = jax.device_put(params, device)
        head = jax.device_put(head, device)
        state = init_adam_state(params, head, MOONS_PATH_LR)
        train_batch_size = int(SYNTHETIC_BATCH_SIZE_BY_QUBITS[int(n_qubits)])
        rng = np.random.default_rng(SYNTHETIC_BASE_SEEDS['moons'] + MOONS_PATH_TRAIN_SEED_OFFSET + int(n_qubits))
        perm = rng.permutation(X_train.shape[0])
        cursor = 0
        latest_train_loss = np.nan

        def record(step):
            pr, gr, glob, heatmap = _moons_eval_gradient_stats(
                params=params,
                head=head,
                meta=meta,
                eval_batches=eval_batches,
                n_qubits=int(n_qubits),
                step=int(step),
                device=device,
            )
            glob['train_loss'] = float(latest_train_loss) if np.isfinite(latest_train_loss) else np.nan
            glob['device'] = str(device)
            glob['used_cpu_fallback'] = bool(used_cpu_fallback)
            glob['eval_batches'] = int(config['eval_batches'])
            glob['eval_batch_size'] = int(config['eval_batch_size'])
            param_rows.extend(pr)
            group_rows.extend(gr)
            global_rows.append(glob)
            heatmap_data[(int(n_qubits), int(step))] = heatmap
            print(
                f"moons path | n={int(n_qubits):2d} | step={int(step):3d} | "
                f"eval_loss={glob['eval_loss_mean']:.4f} | "
                f"grad_norm={glob['grad_norm_mean']:.3e} | "
                f"mean_abs={glob['mean_param_abs_grad']:.3e} | snr={glob['mean_param_snr']:.3e}"
            )

        if 0 in snapshot_steps:
            record(0)
        for step_idx in range(1, max(snapshot_steps) + 1):
            if cursor + train_batch_size > X_train.shape[0]:
                perm = rng.permutation(X_train.shape[0])
                cursor = 0
            batch_idx = perm[cursor:cursor + train_batch_size]
            cursor += train_batch_size
            X_batch = jnp.asarray(X_train[batch_idx], dtype=SYNTHETIC_DTYPE)
            y_batch = jnp.asarray(y_train[batch_idx], dtype=SYNTHETIC_DTYPE)
            X_batch = jax.device_put(X_batch, device)
            y_batch = jax.device_put(y_batch, device)
            params, head, state, lval = train_synthetic_step(
                params,
                head,
                state,
                X_batch,
                y_batch,
                meta,
                optimizer=MOONS_PATH_OPTIMIZER,
                microbatch_size=None,
            )
            block_tree_until_ready_local((params, head, lval))
            latest_train_loss = float(jax.device_get(lval))
            if step_idx in snapshot_steps:
                record(step_idx)

    recorded = sorted({row['step'] for row in global_rows})
    expected = list(snapshot_steps)
    if recorded != expected:
        raise RuntimeError(f'Checkpoint mismatch: recorded {recorded}, expected {expected}')
    if not param_rows or not group_rows or not global_rows or not heatmap_data:
        raise RuntimeError('Moons path diagnostics produced empty result tables.')
    return param_rows, group_rows, global_rows, heatmap_data, device


def print_moons_path_global_table(rows):
    if not rows:
        print('No moons path diagnostic rows to summarize.')
        return
    header = 'qubits | step | train loss | eval loss | norm mean | norm std | mean var | mean snr | mean abs'
    print(header)
    print('-' * len(header))
    for row in sorted(rows, key=lambda r: (r['n_qubits'], r['step'])):
        train_loss = row['train_loss']
        train_loss_text = 'nan' if not np.isfinite(train_loss) else f'{train_loss:.4f}'
        print(
            f"{row['n_qubits']:>6d} | {row['step']:>4d} | {train_loss_text:>10s} | "
            f"{row['eval_loss_mean']:.4f}±{row['eval_loss_std']:.4f} | "
            f"{row['grad_norm_mean']:.3e} | {row['grad_norm_std']:.3e} | "
            f"{row['mean_param_var']:.3e} | {row['mean_param_snr']:.3e} | {row['mean_param_abs_grad']:.3e}"
        )
