"""Autodiff and cached gradients for BMERA regression loss."""

from __future__ import annotations

from typing import Any, Dict, List
import jax
import jax.numpy as jnp
import numpy as np
from .gates import H, PaulyZ
from .simulator import (
    _apply_block_sequence_local,
    _build_block_tensors,
    _pack_params,
    _supervised_inputs,
    _trace_tensor,
    forward_batch,
    forward_single_with_cache,
)
from .topology import _get_meta


_AUTODIFF_GRAD_FNS_CACHE: Dict[tuple[int, int], Any] = {}


_AUTODIFF_MASKED_GRAD_FNS_CACHE: Dict[tuple[int, int, int], Any] = {}


def _reshape_grad_like_params(grad_flat: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    if params.ndim == 3:
        return grad_flat.reshape(params.shape)
    return grad_flat


def _add_cotangent(cot_nodes: List[Any], idx: int, value: jnp.ndarray):
    if cot_nodes[idx] is None:
        cot_nodes[idx] = value
    else:
        cot_nodes[idx] = cot_nodes[idx] + value


def _params_grad_from_cached_forward(
    params: jnp.ndarray,
    meta: Dict[str, Any],
    cache: Dict[str, Any],
    cot_expvals: jnp.ndarray,
) -> jnp.ndarray:
    plan = meta["contraction_plan"]
    nodes = plan["nodes"]
    outputs = plan["outputs"]
    node_outputs = cache["node_outputs"]
    blocks = cache["blocks"]
    blocks_h = jax.vmap(H)(blocks)
    z = PaulyZ()

    cot_nodes: List[Any] = [None] * len(nodes)
    for out in outputs:
        q = int(out["qubit"])
        root = int(out["root"])
        trace_pairs = out["trace_pairs"]

        def observable_from_root(root_tensor: jnp.ndarray):
            rho = _trace_tensor(root_tensor, trace_pairs)
            return jnp.real(jnp.einsum(z, [0, 1], rho, [0, 1], []))

        _, pb = jax.vjp(observable_from_root, node_outputs[root])
        cot_root = pb(cot_expvals[q])[0]
        _add_cotangent(cot_nodes, root, cot_root)

    grad_blocks = jnp.zeros_like(blocks)
    grad_blocks_h = jnp.zeros_like(blocks_h)

    for node_id in range(len(nodes) - 1, -1, -1):
        cot_out = cot_nodes[node_id]
        if cot_out is None:
            continue

        node = nodes[node_id]
        kind = node["kind"]
        if kind in ("layer0", "empty"):
            continue

        param_indices = node["block_param_indices"]
        idx = jnp.array(param_indices, dtype=jnp.int32)
        blocks_local = blocks[idx]
        blocks_h_local = blocks_h[idx]
        block_positions = node["block_positions"]
        block_apply_specs = node["block_apply_specs"]

        if kind == "layer1":
            left = node_outputs[node["left"]]
            right = node_outputs[node["right"]]

            def local_fn(left_t, right_t, local_blocks, local_blocks_h):
                rho = jnp.tensordot(left_t, right_t, axes=0)
                return _apply_block_sequence_local(rho, local_blocks, local_blocks_h, block_positions, block_apply_specs)

            _, pb = jax.vjp(local_fn, left, right, blocks_local, blocks_h_local)
            cot_left, cot_right, cot_blocks_local, cot_blocks_h_local = pb(cot_out)
            _add_cotangent(cot_nodes, node["left"], cot_left)
            _add_cotangent(cot_nodes, node["right"], cot_right)
            grad_blocks = grad_blocks.at[idx].add(cot_blocks_local)
            grad_blocks_h = grad_blocks_h.at[idx].add(cot_blocks_h_local)
            continue

        if kind == "pre_green":
            left = node_outputs[node["left"]]
            right = node_outputs[node["right"]]
            trace_pairs = node["trace_pairs"]
            perm = node["perm"]

            def local_fn(left_t, right_t, local_blocks, local_blocks_h):
                rho = jnp.tensordot(left_t, right_t, axes=0)
                rho = _trace_tensor(rho, trace_pairs)
                rho = jnp.transpose(rho, perm)
                return _apply_block_sequence_local(rho, local_blocks, local_blocks_h, block_positions, block_apply_specs)

            _, pb = jax.vjp(local_fn, left, right, blocks_local, blocks_h_local)
            cot_left, cot_right, cot_blocks_local, cot_blocks_h_local = pb(cot_out)
            _add_cotangent(cot_nodes, node["left"], cot_left)
            _add_cotangent(cot_nodes, node["right"], cot_right)
            grad_blocks = grad_blocks.at[idx].add(cot_blocks_local)
            grad_blocks_h = grad_blocks_h.at[idx].add(cot_blocks_h_local)
            continue

        main = node_outputs[node["main"]]
        trace_pairs = node["trace_pairs"]
        perm = node["perm"]

        def local_fn(main_t, local_blocks, local_blocks_h):
            rho = _trace_tensor(main_t, trace_pairs)
            rho = jnp.transpose(rho, perm)
            return _apply_block_sequence_local(rho, local_blocks, local_blocks_h, block_positions, block_apply_specs)

        _, pb = jax.vjp(local_fn, main, blocks_local, blocks_h_local)
        cot_main, cot_blocks_local, cot_blocks_h_local = pb(cot_out)
        _add_cotangent(cot_nodes, node["main"], cot_main)
        grad_blocks = grad_blocks.at[idx].add(cot_blocks_local)
        grad_blocks_h = grad_blocks_h.at[idx].add(cot_blocks_h_local)

    params_flat = _pack_params(params)
    _, pb_blocks = jax.vjp(lambda p: _build_block_tensors(int(meta["type"]), p), params_flat)
    grad_params_flat = pb_blocks((grad_blocks, grad_blocks_h))[0]
    return _reshape_grad_like_params(grad_params_flat, params)


def loss_and_grad_cached(
    params: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    meta: Dict[str, Any],
    head_weights: jnp.ndarray,
):
    params, X, y, head_weights = _supervised_inputs(params, X, y, meta, head_weights)
    batch_size = int(X.shape[0])

    loss_total = jnp.asarray(0.0, dtype=head_weights.dtype)
    grad_params_total = jnp.zeros_like(params)
    grad_head_total = jnp.zeros_like(head_weights)

    for i in range(batch_size):
        expvals, cache = forward_single_with_cache(params, X[i], meta)
        pred = jnp.dot(expvals, head_weights)
        err = pred - y[i]
        coeff = 2.0 * err

        loss_total = loss_total + err * err
        grad_head_total = grad_head_total + coeff * expvals
        grad_params_total = grad_params_total + _params_grad_from_cached_forward(
            params=params,
            meta=meta,
            cache=cache,
            cot_expvals=coeff * head_weights,
        )

    scale = jnp.asarray(1.0 / batch_size, dtype=head_weights.dtype)
    return loss_total * scale, grad_params_total * scale, grad_head_total * scale


def loss(params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, meta: Dict[str, Any], head_weights: jnp.ndarray) -> jnp.ndarray:
    params, X, y, head_weights = _supervised_inputs(params, X, y, meta, head_weights)
    preds_all = forward_batch(params, X, meta)
    preds = preds_all @ head_weights
    err = preds - y
    return jnp.mean(err * err)


def _masked_loss(
    params: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    meta: Dict[str, Any],
    head_weights: jnp.ndarray,
) -> jnp.ndarray:
    preds_all = forward_batch(params, X, meta)
    preds = preds_all @ head_weights
    err = preds - y
    mask = jnp.asarray(mask, dtype=err.dtype)
    denom = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=err.dtype))
    return jnp.sum(mask * err * err) / denom


def _get_autodiff_grad_fn(n_qubits: int, type: int):
    key = (int(n_qubits), int(type))
    if key not in _AUTODIFF_GRAD_FNS_CACHE:
        meta = _get_meta(n_qubits, type)
        _AUTODIFF_GRAD_FNS_CACHE[key] = jax.jit(
            lambda p, X, y, h: jax.value_and_grad(loss, (0, 4))(p, X, y, meta, h)
        )
    return _AUTODIFF_GRAD_FNS_CACHE[key]


def _get_autodiff_masked_grad_fn(n_qubits: int, type: int, microbatch_size: int):
    key = (int(n_qubits), int(type), int(microbatch_size))
    if key not in _AUTODIFF_MASKED_GRAD_FNS_CACHE:
        meta = _get_meta(n_qubits, type)
        _AUTODIFF_MASKED_GRAD_FNS_CACHE[key] = jax.jit(
            lambda p, X, y, m, h: jax.value_and_grad(
                lambda params_, X_, y_, mask_, head_: _masked_loss(params_, X_, y_, mask_, meta, head_),
                (0, 4),
            )(p, X, y, m, h)
        )
    return _AUTODIFF_MASKED_GRAD_FNS_CACHE[key]


def _effective_microbatch_size(batch_size: int, microbatch_size: int | None) -> int:
    if microbatch_size is None:
        return int(batch_size)
    microbatch_size = int(microbatch_size)
    if microbatch_size <= 0 or microbatch_size >= int(batch_size):
        return int(batch_size)
    return microbatch_size


def _loss_and_grad_autodiff_full(
    params: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    meta: Dict[str, Any],
    head_weights: jnp.ndarray,
):
    params, X, y, head_weights = _supervised_inputs(params, X, y, meta, head_weights)
    grad_fn = _get_autodiff_grad_fn(meta["n_qubits"], meta["type"])
    lval, (grad_params, grad_head) = grad_fn(params, X, y, head_weights)
    return lval, grad_params, grad_head


def loss_and_grad_autodiff_microbatched(
    params: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    meta: Dict[str, Any],
    head_weights: jnp.ndarray,
    microbatch_size: int | None,
):
    params, X, y, head_weights = _supervised_inputs(params, X, y, meta, head_weights)
    batch_size = int(X.shape[0])
    effective_microbatch = _effective_microbatch_size(batch_size, microbatch_size)
    if effective_microbatch >= batch_size:
        return _loss_and_grad_autodiff_full(params, X, y, meta, head_weights)

    n_qubits = int(X.shape[1])
    n_micro = int(np.ceil(batch_size / effective_microbatch))
    padded_batch = int(n_micro * effective_microbatch)
    pad = int(padded_batch - batch_size)

    if pad > 0:
        X_pad = jnp.pad(X, ((0, pad), (0, 0)))
        y_pad = jnp.pad(y, ((0, pad),))
        mask = jnp.pad(jnp.ones((batch_size,), dtype=X.dtype), ((0, pad),))
    else:
        X_pad = X
        y_pad = y
        mask = jnp.ones((batch_size,), dtype=X.dtype)

    grad_fn = _get_autodiff_masked_grad_fn(meta["n_qubits"], meta["type"], effective_microbatch)
    batch_scale = jnp.asarray(float(batch_size), dtype=X.dtype)

    def body(i, carry):
        loss_total, grad_params_total, grad_head_total = carry
        start = i * effective_microbatch
        Xi = jax.lax.dynamic_slice(X_pad, (start, 0), (effective_microbatch, n_qubits))
        yi = jax.lax.dynamic_slice(y_pad, (start,), (effective_microbatch,))
        mi = jax.lax.dynamic_slice(mask, (start,), (effective_microbatch,))
        lval, (grad_params, grad_head) = grad_fn(params, Xi, yi, mi, head_weights)
        chunk_weight = jnp.sum(mi)
        coeff = chunk_weight / batch_scale
        return (
            loss_total + coeff * lval,
            grad_params_total + coeff * grad_params,
            grad_head_total + coeff * grad_head,
        )

    init = (
        jnp.asarray(0.0, dtype=X.dtype),
        jnp.zeros_like(params),
        jnp.zeros_like(head_weights),
    )
    return jax.lax.fori_loop(0, n_micro, body, init)


def _loss_and_grad_autodiff(
    params: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    meta: Dict[str, Any],
    head_weights: jnp.ndarray,
    microbatch_size: int | None = None,
):
    params, X, y, head_weights = _supervised_inputs(params, X, y, meta, head_weights)
    effective_microbatch = _effective_microbatch_size(int(X.shape[0]), microbatch_size)
    if effective_microbatch >= int(X.shape[0]):
        return _loss_and_grad_autodiff_full(params, X, y, meta, head_weights)
    return loss_and_grad_autodiff_microbatched(params, X, y, meta, head_weights, effective_microbatch)
