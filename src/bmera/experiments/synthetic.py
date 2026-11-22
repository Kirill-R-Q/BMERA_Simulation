"""Synthetic classification datasets and training."""

import jax
import jax.numpy as jnp
import numpy as np
from bmera import build_bmera_meta, forward_batch, loss_and_grad_autodiff_microbatched, train_step

# Synthetic fixed-information dataset benchmark helpers.
# Pure NumPy generators are used intentionally so this section has no sklearn dependency.
import json
from pathlib import Path


SYNTHETIC_DATASETS = ('moons', 'circles', 'xor', 'parity')
SYNTHETIC_QUBITS = (8, 16)
SYNTHETIC_OPTIMIZERS = ('sgd', 'adam')
SYNTHETIC_BATCH_SIZE_BY_QUBITS = {
    8: 2048,
    16: 1024,
    32: 1024,
    64: 128,
}
SYNTHETIC_N_TRAIN = 4096
SYNTHETIC_N_VAL = 1024
SYNTHETIC_N_TEST = 2048
SYNTHETIC_STEPS = 2000
SYNTHETIC_LR = 1e-2
SYNTHETIC_LR_BY_OPTIMIZER = {
    'sgd': 1e-2,
    'adam': 1e-2,
}
SYNTHETIC_LOG_EVERY = 200
SYNTHETIC_REPEATS = 3
SYNTHETIC_TYPE = 1
SYNTHETIC_DTYPE = jnp.float32
SYNTHETIC_SAVE_WEIGHTS = True
SYNTHETIC_SAVE_DIR = Path('synthetic_bmera_runs')
SYNTHETIC_BASE_SEEDS = {
    'moons': 202401,
    'circles': 202402,
    'xor': 202403,
    'parity': 202404,
}


def _shuffle_dataset(X, y, rng):
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def _make_moons(n_samples, rng, noise=0.10):
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer
    theta_outer = rng.uniform(0.0, np.pi, size=n_outer)
    theta_inner = rng.uniform(0.0, np.pi, size=n_inner)
    outer = np.stack([np.cos(theta_outer), np.sin(theta_outer)], axis=1)
    inner = np.stack([1.0 - np.cos(theta_inner), 0.5 - np.sin(theta_inner)], axis=1)
    X = np.concatenate([outer, inner], axis=0)
    y = np.concatenate([
        -np.ones(n_outer, dtype=np.float32),
        np.ones(n_inner, dtype=np.float32),
    ])
    X = X + rng.normal(0.0, noise, size=X.shape)
    return _shuffle_dataset(X.astype(np.float32), y, rng)


def _make_circles(n_samples, rng, noise=0.06, inner_radius=0.45):
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner
    theta_inner = rng.uniform(0.0, 2.0 * np.pi, size=n_inner)
    theta_outer = rng.uniform(0.0, 2.0 * np.pi, size=n_outer)
    inner = inner_radius * np.stack([np.cos(theta_inner), np.sin(theta_inner)], axis=1)
    outer = np.stack([np.cos(theta_outer), np.sin(theta_outer)], axis=1)
    X = np.concatenate([inner, outer], axis=0)
    y = np.concatenate([
        -np.ones(n_inner, dtype=np.float32),
        np.ones(n_outer, dtype=np.float32),
    ])
    X = X + rng.normal(0.0, noise, size=X.shape)
    return _shuffle_dataset(X.astype(np.float32), y, rng)


def _make_xor(n_samples, rng, noise=0.04):
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 2)).astype(np.float32)
    y = np.where(X[:, 0] * X[:, 1] >= 0.0, 1.0, -1.0).astype(np.float32)
    X = X + rng.normal(0.0, noise, size=X.shape).astype(np.float32)
    return _shuffle_dataset(X, y, rng)


def _make_parity(n_samples, rng):
    X = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_samples, 8)).astype(np.float32)
    y = np.prod(X[:, :4], axis=1).astype(np.float32)
    return _shuffle_dataset(X, y, rng)


def make_synthetic_base_dataset(name, n_train=SYNTHETIC_N_TRAIN, n_val=SYNTHETIC_N_VAL, n_test=SYNTHETIC_N_TEST):
    if name not in SYNTHETIC_DATASETS:
        raise ValueError(f'Unknown synthetic dataset {name!r}. Expected one of {SYNTHETIC_DATASETS}.')
    rng = np.random.default_rng(SYNTHETIC_BASE_SEEDS[name])
    total = int(n_train + n_val + n_test)
    if name == 'moons':
        X, y = _make_moons(total, rng)
    elif name == 'circles':
        X, y = _make_circles(total, rng)
    elif name == 'xor':
        X, y = _make_xor(total, rng)
    else:
        X, y = _make_parity(total, rng)

    return {
        'train': (X[:n_train], y[:n_train]),
        'val': (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        'test': (X[n_train + n_val:], y[n_train + n_val:]),
    }


def fit_synthetic_preprocessor(X_train):
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return {'mean': mean, 'std': std}


def transform_synthetic_features(X, preprocessor):
    X_std = (X.astype(np.float32) - preprocessor['mean']) / preprocessor['std']
    X_clip = np.clip(X_std, -3.0, 3.0)
    return (X_clip / 6.0).astype(np.float32)


def embed_synthetic_features(X_base, n_qubits):
    n_qubits = int(n_qubits)
    base_dim = int(X_base.shape[1])
    if base_dim > n_qubits:
        raise ValueError(f'Base feature dimension {base_dim} does not fit into {n_qubits} qubits.')
    X_embed = np.zeros((X_base.shape[0], n_qubits), dtype=np.float32)
    X_embed[:, :base_dim] = X_base.astype(np.float32)
    return X_embed


def prepare_synthetic_dataset(name, n_qubits):
    raw = make_synthetic_base_dataset(name)
    preprocessor = fit_synthetic_preprocessor(raw['train'][0])
    prepared = {'name': name, 'n_qubits': int(n_qubits), 'preprocessor': preprocessor}
    for split_name, (X_raw, y) in raw.items():
        X_base = transform_synthetic_features(X_raw, preprocessor)
        prepared[split_name] = (embed_synthetic_features(X_base, n_qubits), y.astype(np.float32))
    return prepared


def assert_synthetic_dataset_ok(prepared):
    n_qubits = int(prepared['n_qubits'])
    base_dim = 8 if prepared['name'] == 'parity' else 2
    for split_name in ('train', 'val', 'test'):
        X, y = prepared[split_name]
        assert X.shape[1] == n_qubits, (split_name, X.shape, n_qubits)
        assert set(np.unique(y).tolist()) <= {-1.0, 1.0}, (split_name, np.unique(y))
        if n_qubits > base_dim:
            assert np.all(X[:, base_dim:] == 0.0), (split_name, 'padding is not exactly zero')
    assert prepared['preprocessor']['mean'].shape[1] == base_dim
    assert prepared['preprocessor']['std'].shape[1] == base_dim


def synthetic_target_device():
    devices = [device for device in jax.devices() if device.platform in ('gpu', 'cuda', 'rocm')]
    return devices[0] if devices else None


def block_tree_until_ready_local(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, 'block_until_ready'):
            leaf.block_until_ready()


def predict_bmera_in_batches(params, head, X, meta, batch_size, device=None):
    preds = []
    for start in range(0, X.shape[0], int(batch_size)):
        X_chunk = jnp.asarray(X[start:start + int(batch_size)], dtype=SYNTHETIC_DTYPE)
        if device is not None:
            X_chunk = jax.device_put(X_chunk, device)
        pred_chunk = forward_batch(params, X_chunk, meta) @ head
        pred_chunk.block_until_ready()
        preds.append(np.asarray(pred_chunk))
    return np.concatenate(preds, axis=0)


def evaluate_bmera_classifier(params, head, X, y, meta, batch_size, device=None):
    preds = predict_bmera_in_batches(params, head, X, meta, batch_size, device=device)
    y_np = np.asarray(y, dtype=np.float32)
    mse = float(np.mean((preds - y_np) ** 2))
    labels = np.where(preds >= 0.0, 1.0, -1.0).astype(np.float32)
    accuracy = float(np.mean(labels == y_np))
    return {'mse': mse, 'accuracy': accuracy, 'preds': preds}


def initialize_bmera_params(n_qubits, repeat, dataset_name, type_=SYNTHETIC_TYPE):
    meta = build_bmera_meta(int(n_qubits), int(type_))
    seed = SYNTHETIC_BASE_SEEDS[dataset_name] + 1009 * int(n_qubits) + 9176 * int(repeat)
    rng = np.random.default_rng(seed)
    param_shape = (int(meta['depth']) * int(n_qubits), int(meta['n_per_block']))
    params = jnp.asarray(rng.uniform(-0.4, 0.4, size=param_shape), dtype=SYNTHETIC_DTYPE)
    head = jnp.asarray(rng.uniform(-0.4, 0.4, size=(int(n_qubits),)), dtype=SYNTHETIC_DTYPE)
    return meta, params, head


def init_adam_state(params, head, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    return {
        'lr': float(lr),
        'step': 0,
        'beta1': float(beta1),
        'beta2': float(beta2),
        'eps': float(eps),
        'm_params': jnp.zeros_like(params),
        'v_params': jnp.zeros_like(params),
        'm_head': jnp.zeros_like(head),
        'v_head': jnp.zeros_like(head),
    }


def adam_update(params, head, state, grad_params, grad_head):
    step = int(state['step']) + 1
    beta1 = jnp.asarray(state.get('beta1', 0.9), dtype=params.dtype)
    beta2 = jnp.asarray(state.get('beta2', 0.999), dtype=params.dtype)
    eps = jnp.asarray(state.get('eps', 1e-8), dtype=params.dtype)
    lr = jnp.asarray(state.get('lr', 1e-3), dtype=params.dtype)

    m_params = beta1 * state['m_params'] + (1.0 - beta1) * grad_params
    v_params = beta2 * state['v_params'] + (1.0 - beta2) * jnp.square(grad_params)
    m_head = beta1 * state['m_head'] + (1.0 - beta1) * grad_head
    v_head = beta2 * state['v_head'] + (1.0 - beta2) * jnp.square(grad_head)

    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    params = params - lr * (m_params / bias_correction1) / (jnp.sqrt(v_params / bias_correction2) + eps)
    head = head - lr * (m_head / bias_correction1) / (jnp.sqrt(v_head / bias_correction2) + eps)

    new_state = {
        **state,
        'step': step,
        'm_params': m_params,
        'v_params': v_params,
        'm_head': m_head,
        'v_head': v_head,
    }
    return params, head, new_state


def train_synthetic_step(params, head, state, X_batch, y_batch, meta, optimizer, microbatch_size=None):
    optimizer = optimizer.lower()
    if optimizer == 'sgd':
        return train_step(
            params,
            state,
            X_batch,
            y_batch,
            meta,
            head,
            microbatch_size=microbatch_size,
        )
    if optimizer == 'adam':
        lval, grad_params, grad_head = loss_and_grad_autodiff_microbatched(
            params,
            X_batch,
            y_batch,
            meta,
            head,
            microbatch_size,
        )
        params, head, state = adam_update(params, head, state, grad_params, grad_head)
        return params, head, state, lval
    raise ValueError(f'Unknown optimizer {optimizer!r}. Expected one of {SYNTHETIC_OPTIMIZERS}.')


def synthetic_artifact_stem(result):
    dataset = str(result['dataset']).replace('/', '_')
    optimizer = str(result['optimizer']).replace('/', '_')
    return (
        f"{dataset}_n{int(result['n_qubits'])}_{optimizer}_"
        f"repeat{int(result['repeat'])}_steps{int(result['steps'])}"
    )


def synthetic_result_metadata(result):
    excluded = {'params', 'head_weights', 'losses'}
    metadata = {'artifact_version': 1}
    for key, value in result.items():
        if key in excluded:
            continue
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        metadata[key] = value
    return metadata


def save_synthetic_training_artifacts(result, save_dir=SYNTHETIC_SAVE_DIR):
    if result['status'] != 'ok':
        return None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = synthetic_artifact_stem(result)
    weights_path = save_dir / f'{stem}.npz'
    metadata_path = save_dir / f'{stem}.json'
    metadata = synthetic_result_metadata(result)
    metadata['weights_path'] = str(weights_path)
    metadata['metadata_path'] = str(metadata_path)

    metadata_json = json.dumps(metadata, indent=2, sort_keys=True)
    np.savez_compressed(
        weights_path,
        params=np.asarray(result['params']),
        head_weights=np.asarray(result['head_weights']),
        losses=np.asarray(result['losses'], dtype=np.float32),
        metadata_json=np.asarray(metadata_json),
    )
    metadata_path.write_text(metadata_json + '\n')
    return {'weights_path': str(weights_path), 'metadata_path': str(metadata_path)}


def train_synthetic_bmera_once(dataset_name, n_qubits, repeat, steps=SYNTHETIC_STEPS, lr=None, optimizer='sgd', log_every=SYNTHETIC_LOG_EVERY, save_weights=SYNTHETIC_SAVE_WEIGHTS, save_dir=SYNTHETIC_SAVE_DIR):
    optimizer = optimizer.lower()
    if optimizer not in SYNTHETIC_OPTIMIZERS:
        raise ValueError(f'Unknown optimizer {optimizer!r}. Expected one of {SYNTHETIC_OPTIMIZERS}.')
    if lr is None:
        lr = SYNTHETIC_LR_BY_OPTIMIZER[optimizer]
    batch_size = int(SYNTHETIC_BATCH_SIZE_BY_QUBITS[int(n_qubits)])
    prepared = prepare_synthetic_dataset(dataset_name, n_qubits)
    assert_synthetic_dataset_ok(prepared)
    X_train, y_train = prepared['train']
    device = synthetic_target_device()
    meta, params, head = initialize_bmera_params(n_qubits, repeat, dataset_name)
    if device is not None:
        params = jax.device_put(params, device)
        head = jax.device_put(head, device)

    if optimizer == 'adam':
        state = init_adam_state(params, head, lr)
    else:
        state = {'lr': float(lr), 'step': 0}
    losses = []
    rng = np.random.default_rng(SYNTHETIC_BASE_SEEDS[dataset_name] + 50021 * int(n_qubits) + 1237 * int(repeat))
    perm = rng.permutation(X_train.shape[0])
    cursor = 0

    for step_idx in range(int(steps)):
        if cursor + batch_size > X_train.shape[0]:
            perm = rng.permutation(X_train.shape[0])
            cursor = 0
        batch_idx = perm[cursor:cursor + batch_size]
        cursor += batch_size
        X_batch = jnp.asarray(X_train[batch_idx], dtype=SYNTHETIC_DTYPE)
        y_batch = jnp.asarray(y_train[batch_idx], dtype=SYNTHETIC_DTYPE)
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
            optimizer=optimizer,
            microbatch_size=None,
        )
        block_tree_until_ready_local((params, head, lval))
        current_loss = float(lval)
        losses.append(current_loss)
        if log_every and ((step_idx + 1) == 1 or (step_idx + 1) % int(log_every) == 0 or (step_idx + 1) == int(steps)):
            print(
                f'  train | optimizer={optimizer:4s} | dataset={dataset_name:7s} | '
                f'n={int(n_qubits):2d} | repeat={int(repeat)} | step={step_idx + 1:4d}/{int(steps)} | '
                f'loss={current_loss:.6f}'
            )

    metrics = {}
    for split_name in ('train', 'val', 'test'):
        X_split, y_split = prepared[split_name]
        split_metrics = evaluate_bmera_classifier(params, head, X_split, y_split, meta, batch_size, device=device)
        metrics[f'{split_name}_mse'] = split_metrics['mse']
        metrics[f'{split_name}_accuracy'] = split_metrics['accuracy']

    epochs = float(int(steps) * batch_size / X_train.shape[0])
    result = {
        'dataset': dataset_name,
        'n_qubits': int(n_qubits),
        'optimizer': optimizer,
        'repeat': int(repeat),
        'n_train': int(prepared['train'][0].shape[0]),
        'n_val': int(prepared['val'][0].shape[0]),
        'n_test': int(prepared['test'][0].shape[0]),
        'train_batch_size': batch_size,
        'steps': int(steps),
        'epochs': epochs,
        'lr': float(lr),
        'log_every': int(log_every) if log_every else None,
        'microbatch_size': None,
        'bmera_type': int(SYNTHETIC_TYPE),
        'dtype': str(SYNTHETIC_DTYPE),
        'base_seed': int(SYNTHETIC_BASE_SEEDS[dataset_name]),
        'init_seed': int(SYNTHETIC_BASE_SEEDS[dataset_name] + 1009 * int(n_qubits) + 9176 * int(repeat)),
        'shuffle_seed': int(SYNTHETIC_BASE_SEEDS[dataset_name] + 50021 * int(n_qubits) + 1237 * int(repeat)),
        'train_mse': metrics['train_mse'],
        'val_mse': metrics['val_mse'],
        'test_mse': metrics['test_mse'],
        'train_accuracy': metrics['train_accuracy'],
        'val_accuracy': metrics['val_accuracy'],
        'test_accuracy': metrics['test_accuracy'],
        'final_loss': float(losses[-1]) if losses else np.nan,
        'losses': np.asarray(losses, dtype=np.float32),
        'params': params,
        'head_weights': head,
        'weights_path': None,
        'metadata_path': None,
        'status': 'ok',
        'error': None,
    }

    if save_weights:
        artifact_paths = save_synthetic_training_artifacts(result, save_dir=save_dir)
        if artifact_paths is not None:
            result.update(artifact_paths)
            print(f"  saved | weights={result['weights_path']} | metadata={result['metadata_path']}")

    return result


def failed_synthetic_result(dataset_name, n_qubits, repeat, exc, steps=SYNTHETIC_STEPS, lr=None, optimizer='sgd', log_every=SYNTHETIC_LOG_EVERY):
    optimizer = optimizer.lower()
    if lr is None:
        lr = SYNTHETIC_LR_BY_OPTIMIZER.get(optimizer, SYNTHETIC_LR)
    return {
        'dataset': dataset_name,
        'n_qubits': int(n_qubits),
        'optimizer': optimizer,
        'repeat': int(repeat),
        'n_train': int(SYNTHETIC_N_TRAIN),
        'n_val': int(SYNTHETIC_N_VAL),
        'n_test': int(SYNTHETIC_N_TEST),
        'train_batch_size': int(SYNTHETIC_BATCH_SIZE_BY_QUBITS[int(n_qubits)]),
        'steps': int(steps),
        'epochs': float(int(steps) * int(SYNTHETIC_BATCH_SIZE_BY_QUBITS[int(n_qubits)]) / SYNTHETIC_N_TRAIN),
        'lr': float(lr),
        'log_every': int(log_every) if log_every else None,
        'microbatch_size': None,
        'bmera_type': int(SYNTHETIC_TYPE),
        'dtype': str(SYNTHETIC_DTYPE),
        'base_seed': int(SYNTHETIC_BASE_SEEDS[dataset_name]),
        'init_seed': int(SYNTHETIC_BASE_SEEDS[dataset_name] + 1009 * int(n_qubits) + 9176 * int(repeat)),
        'shuffle_seed': int(SYNTHETIC_BASE_SEEDS[dataset_name] + 50021 * int(n_qubits) + 1237 * int(repeat)),
        'train_mse': np.nan,
        'val_mse': np.nan,
        'test_mse': np.nan,
        'train_accuracy': np.nan,
        'val_accuracy': np.nan,
        'test_accuracy': np.nan,
        'final_loss': np.nan,
        'losses': np.asarray([], dtype=np.float32),
        'params': None,
        'head_weights': None,
        'weights_path': None,
        'metadata_path': None,
        'status': 'failed',
        'error': repr(exc),
    }


def run_synthetic_bmera_sweep(datasets=SYNTHETIC_DATASETS, qubits=SYNTHETIC_QUBITS, repeats=SYNTHETIC_REPEATS, optimizers=SYNTHETIC_OPTIMIZERS, save_weights=SYNTHETIC_SAVE_WEIGHTS, save_dir=SYNTHETIC_SAVE_DIR):
    results = []
    for dataset_name in datasets:
        for n_qubits in qubits:
            for optimizer in optimizers:
                optimizer = optimizer.lower()
                lr = SYNTHETIC_LR_BY_OPTIMIZER[optimizer]
                for repeat in range(int(repeats)):
                    print(
                        f'Running dataset={dataset_name}, n_qubits={n_qubits}, optimizer={optimizer}, '
                        f'repeat={repeat}, batch={SYNTHETIC_BATCH_SIZE_BY_QUBITS[int(n_qubits)]}, lr={lr}'
                    )
                    try:
                        result = train_synthetic_bmera_once(
                            dataset_name,
                            n_qubits,
                            repeat,
                            lr=lr,
                            optimizer=optimizer,
                            save_weights=save_weights,
                            save_dir=save_dir,
                        )
                        print(
                            f"  ok | optimizer={optimizer} | val_acc={result['val_accuracy']:.3f} | "
                            f"test_acc={result['test_accuracy']:.3f} | final_loss={result['final_loss']:.4f}"
                        )
                    except Exception as exc:
                        result = failed_synthetic_result(dataset_name, n_qubits, repeat, exc, lr=lr, optimizer=optimizer)
                        print(f'  failed: {exc!r}')
                    results.append(result)
    return results


def select_best_synthetic_results(results):
    best = {}
    for result in results:
        if result['status'] != 'ok':
            continue
        key = (result['dataset'], result['n_qubits'], result['optimizer'])
        current = best.get(key)
        if current is None:
            best[key] = result
            continue
        candidate_score = (result['val_accuracy'], -result['val_mse'])
        current_score = (current['val_accuracy'], -current['val_mse'])
        if candidate_score > current_score:
            best[key] = result
    return best


def print_synthetic_results_table(results):
    best = select_best_synthetic_results(results)
    print('dataset  | qubits | optimizer | repeat | batch | val_acc | test_acc | val_mse | test_mse | status')
    print('-' * 106)
    for dataset_name in SYNTHETIC_DATASETS:
        for n_qubits in SYNTHETIC_QUBITS:
            for optimizer in SYNTHETIC_OPTIMIZERS:
                result = best.get((dataset_name, int(n_qubits), optimizer))
                if result is None:
                    failures = [
                        r for r in results
                        if r['dataset'] == dataset_name and r['n_qubits'] == int(n_qubits) and r.get('optimizer') == optimizer
                    ]
                    status = failures[-1]['status'] if failures else 'missing'
                    print(f'{dataset_name:8s} | {n_qubits:6d} | {optimizer:9s} |      - |     - |       - |        - |       - |        - | {status}')
                    continue
                print(
                    f"{dataset_name:8s} | {n_qubits:6d} | {optimizer:9s} | {result['repeat']:6d} | "
                    f"{result['train_batch_size']:5d} | {result['val_accuracy']:7.3f} | "
                    f"{result['test_accuracy']:8.3f} | {result['val_mse']:7.4f} | "
                    f"{result['test_mse']:8.4f} | {result['status']}"
                )
