"""Synchronized JAX timing and training benchmarks."""

from __future__ import annotations

import time
from typing import List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from .gradients import (
    _effective_microbatch_size,
    _loss_and_grad_autodiff,
    loss,
    loss_and_grad_cached,
)
from .simulator import (
    _cache_nbytes,
    forward_batch,
    forward_single_with_cache,
)
from .topology import _get_meta
from .training import _train_step_autodiff_full, train_step_cached


def _block_tree_until_ready(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def benchmark_callable(fn, repeats: int, warmup: int):
    """Measure the first call and synchronized steady-state calls in milliseconds."""
    t0 = time.perf_counter()
    first = fn()
    _block_tree_until_ready(first)
    compile_and_first_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(max(0, warmup)):
        out = fn()
        _block_tree_until_ready(out)

    samples = []
    for _ in range(max(1, repeats)):
        ts = time.perf_counter()
        out = fn()
        _block_tree_until_ready(out)
        samples.append((time.perf_counter() - ts) * 1000.0)

    return {
        "compile_plus_first_ms": float(compile_and_first_ms),
        "steady_mean_ms": float(np.mean(samples)),
        "steady_std_ms": float(np.std(samples)),
    }


def benchmark_context(device, dtype, repeats, warmup, seed):
    return {
        "device_kind": device.device_kind,
        "device_platform": device.platform,
        "dtype": np.dtype(dtype).name,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "repeats": max(1, repeats),
        "warmup": max(0, warmup),
        "seed": seed,
    }


def _resolve_device(device: str | None):
    if device is None:
        return None
    devices = jax.devices(device)
    if not devices:
        raise ValueError(f"No JAX device found for '{device}'")
    return devices[0]


def benchmark_training_scaling(
    n_qubits_list: List[int],
    batch_sizes: Tuple[int, ...] = (1, 16, 64),
    type: int = 1,
    repeats: int = 5,
    warmup: int = 1,
    dtype=jnp.float32,
    grad_engine: str = "autodiff",
    device: str | None = None,
    seed: int = 0,
    microbatch_size: int | None = None,
):
    if grad_engine not in ("autodiff", "cached"):
        raise ValueError("grad_engine must be 'autodiff' or 'cached'")

    rng = np.random.default_rng(seed)
    target_device = _resolve_device(device)
    rows = []

    for n_qubits in n_qubits_list:
        meta = _get_meta(n_qubits, type)
        n_params_per_block = int(meta["n_per_block"])
        param_shape = (int(meta["depth"]) * n_qubits, n_params_per_block)

        for batch_size in batch_sizes:
            params = jnp.asarray(rng.uniform(-0.5, 0.5, size=param_shape), dtype=dtype)
            X = jnp.asarray(rng.uniform(-0.5, 0.5, size=(batch_size, n_qubits)), dtype=dtype)
            y = jnp.asarray(rng.uniform(-1.0, 1.0, size=(batch_size,)), dtype=dtype)
            head = jnp.asarray(rng.uniform(-0.5, 0.5, size=(n_qubits,)), dtype=dtype)
            opt_state = {"lr": 1e-3, "step": 0}

            if target_device is not None:
                params = jax.device_put(params, target_device)
                X = jax.device_put(X, target_device)
                y = jax.device_put(y, target_device)
                head = jax.device_put(head, target_device)

            fwd_bench = benchmark_callable(lambda: forward_batch(params, X, meta), repeats, warmup)
            loss_bench = benchmark_callable(lambda: loss(params, X, y, meta, head), repeats, warmup)

            if grad_engine == "autodiff":
                grad_fn = lambda: _loss_and_grad_autodiff(
                    params,
                    X,
                    y,
                    meta,
                    head,
                    microbatch_size=microbatch_size,
                )
                train_fn = lambda: _train_step_autodiff_full(
                    params,
                    opt_state,
                    X,
                    y,
                    meta,
                    head,
                    microbatch_size=microbatch_size,
                )
                cache_mb = 0.0
            else:
                grad_fn = lambda: loss_and_grad_cached(params, X, y, meta, head)
                train_fn = lambda: train_step_cached(params, opt_state, X, y, meta, head)
                _, cache_one = forward_single_with_cache(params, X[0], meta)
                cache_mb = _cache_nbytes(cache_one) / (1024.0 * 1024.0)

            grad_bench = benchmark_callable(grad_fn, repeats, warmup)
            train_bench = benchmark_callable(train_fn, repeats, warmup)

            rows.append(
                {
                    "device": device or "default",
                    **benchmark_context(params.device, dtype, repeats, warmup, seed),
                    "grad_engine": grad_engine,
                    "microbatch_size": (None if grad_engine != "autodiff" else microbatch_size),
                    "effective_microbatch_size": (
                        None
                        if grad_engine != "autodiff"
                        else _effective_microbatch_size(int(batch_size), microbatch_size)
                    ),
                    "type": int(type),
                    "n_qubits": int(n_qubits),
                    "batch_size": int(batch_size),
                    "forward_compile_plus_first_ms": fwd_bench["compile_plus_first_ms"],
                    "forward_steady_ms": fwd_bench["steady_mean_ms"],
                    "forward_steady_std_ms": fwd_bench["steady_std_ms"],
                    "loss_compile_plus_first_ms": loss_bench["compile_plus_first_ms"],
                    "loss_steady_ms": loss_bench["steady_mean_ms"],
                    "loss_steady_std_ms": loss_bench["steady_std_ms"],
                    "grad_compile_plus_first_ms": grad_bench["compile_plus_first_ms"],
                    "grad_steady_ms": grad_bench["steady_mean_ms"],
                    "grad_steady_std_ms": grad_bench["steady_std_ms"],
                    "train_compile_plus_first_ms": train_bench["compile_plus_first_ms"],
                    "train_steady_ms": train_bench["steady_mean_ms"],
                    "train_steady_std_ms": train_bench["steady_std_ms"],
                    "cache_mb_per_sample": float(cache_mb),
                }
            )

    return rows
