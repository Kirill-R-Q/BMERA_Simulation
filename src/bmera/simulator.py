"""JAX execution of BMERA contraction plans."""

from __future__ import annotations

from operator import index
from typing import Any, Dict, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from .gates import (
    H,
    InitState,
    PaulyZ,
    createBMERABlock,
)
from .topology import _get_meta


_FORWARD_FNS_CACHE: Dict[tuple[int, int], Tuple[Any, Any]] = {}


_FORWARD_WITH_CACHE_FNS_CACHE: Dict[tuple[int, int], Tuple[Any, Any]] = {}


def _forward_inputs(params, angles, meta, *, batched):
    params, angles = jnp.asarray(params), jnp.asarray(angles)
    layered_shape = (meta["depth"], meta["n_qubits"], meta["n_per_block"])
    flat_shape = (layered_shape[0] * layered_shape[1], layered_shape[2])
    if params.shape not in (flat_shape, layered_shape):
        raise ValueError(f"params must have shape {flat_shape} or {layered_shape}, got {params.shape}")
    if batched:
        if angles.ndim != 2 or angles.shape[1] != meta["n_qubits"]:
            raise ValueError(f"X must have shape (batch_size, {meta['n_qubits']}), got {angles.shape}")
    elif angles.shape != (meta["n_qubits"],):
        raise ValueError(f"x must have shape ({meta['n_qubits']},), got {angles.shape}")
    return params, angles


def _supervised_inputs(params, X, y, meta, head_weights):
    params, X = _forward_inputs(params, X, meta, batched=True)
    y, head_weights = jnp.asarray(y), jnp.asarray(head_weights)
    if not X.shape[0] or y.shape != (X.shape[0],):
        raise ValueError("X and y must contain the same nonzero number of samples, with y rank-1")
    if head_weights.shape != (meta["n_qubits"],):
        raise ValueError(f"head_weights must have shape ({meta['n_qubits']},), got {head_weights.shape}")
    return params, X, y, head_weights


def _pack_params(params: jnp.ndarray) -> jnp.ndarray:
    return params.reshape((-1, params.shape[-1]))


def _validate_qubit(qubit, n_qubits):
    try:
        qubit = index(qubit)
    except TypeError as exc:
        raise ValueError("qubit must be an integer") from exc
    if not 0 <= qubit < n_qubits:
        raise ValueError(f"qubit {qubit} is out of range with n_qubits = {n_qubits}")
    return qubit


def _trace_tensor(tensor: jnp.ndarray, trace_pairs: Tuple[Tuple[int, int], ...]) -> jnp.ndarray:
    out = tensor
    for axis1, axis2 in trace_pairs:
        out = jnp.trace(out, axis1=axis1, axis2=axis2)
    return out


def _apply_unitary_to_density(
    tensor: jnp.ndarray,
    U: jnp.ndarray,
    U_H: jnp.ndarray,
    apply_spec: Dict[str, Tuple[int, ...]],
) -> jnp.ndarray:
    perm = apply_spec["perm"]
    inv_perm = apply_spec["inv_perm"]

    transposed = jnp.transpose(tensor, perm)
    trailing_shape = transposed.shape[:-4]
    rho_blocks = transposed.reshape((-1, 4, 4))

    U_mat = jnp.transpose(U, (0, 2, 1, 3)).reshape((4, 4))
    U_H_mat = jnp.transpose(U_H, (0, 2, 1, 3)).reshape((4, 4))
    updated = jnp.matmul(jnp.matmul(U_mat[jnp.newaxis, :, :], rho_blocks), U_H_mat[jnp.newaxis, :, :])

    restored = updated.reshape(trailing_shape + (2, 2, 2, 2))
    return jnp.transpose(restored, inv_perm)


def _apply_density_block(
    tensor: jnp.ndarray,
    U: jnp.ndarray,
    U_H: jnp.ndarray,
    apply_spec: Dict[str, Tuple[int, ...]],
) -> jnp.ndarray:
    return _apply_unitary_to_density(tensor, U, U_H, apply_spec)


def _apply_block_sequence(
    tensor: jnp.ndarray,
    blocks: jnp.ndarray,
    blocks_h: jnp.ndarray,
    block_positions: Tuple[Tuple[int, int], ...],
    block_apply_specs: Tuple[Dict[str, Tuple[int, ...]], ...],
    block_param_indices: Tuple[int, ...],
) -> jnp.ndarray:
    out = tensor
    for apply_spec, param_idx in zip(block_apply_specs, block_param_indices):
        U = blocks[param_idx]
        U_H = blocks_h[param_idx]
        out = _apply_density_block(out, U, U_H, apply_spec)
    return out


def _apply_block_sequence_local(
    tensor: jnp.ndarray,
    blocks_local: jnp.ndarray,
    blocks_h_local: jnp.ndarray,
    block_positions: Tuple[Tuple[int, int], ...],
    block_apply_specs: Tuple[Dict[str, Tuple[int, ...]], ...],
) -> jnp.ndarray:
    out = tensor
    for i, _ in enumerate(block_positions):
        U = blocks_local[i]
        U_H = blocks_h_local[i]
        out = _apply_density_block(out, U, U_H, block_apply_specs[i])
    return out


def _product_state_density(x: jnp.ndarray, qubits: Tuple[int, ...]) -> jnp.ndarray:
    if len(qubits) == 0:
        cdtype = jnp.complex128 if x.dtype == jnp.float64 else jnp.complex64
        return jnp.asarray(1.0, dtype=cdtype)
    pieces = []
    cdtype = jnp.complex128 if x.dtype == jnp.float64 else jnp.complex64
    for q in qubits:
        state = InitState(x[q]).astype(cdtype)
        pieces.append(jnp.outer(state, jnp.conjugate(state)))
    out = pieces[0]
    for p in pieces[1:]:
        out = jnp.tensordot(out, p, axes=0)
    return out


def _build_block_tensors(type: int, params_flat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if type == 1:
        blocks = jax.vmap(lambda p: createBMERABlock(1, p))(params_flat)
    elif type == 2:
        blocks = jax.vmap(lambda p: createBMERABlock(2, p))(params_flat)
    else:
        blocks = jax.vmap(lambda p: createBMERABlock(3, p))(params_flat)
    blocks_h = jax.vmap(H)(blocks)
    return blocks, blocks_h


def _make_forward_impl(meta: Dict[str, Any]):
    plan = meta["contraction_plan"]
    nodes = plan["nodes"]
    outputs = plan["outputs"]
    bmera_type = int(meta["type"])

    def _forward_impl(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        params_flat = _pack_params(params)
        blocks, blocks_h = _build_block_tensors(bmera_type, params_flat)

        tensors: List[Any] = [None] * len(nodes)
        for i, node in enumerate(nodes):
            kind = node["kind"]
            if kind == "empty":
                cdtype = blocks.dtype
                tensors[i] = jnp.asarray(1.0, dtype=cdtype)
                continue

            if kind == "layer0":
                tensors[i] = _product_state_density(x, node["qubits"])
                continue

            if kind == "layer1":
                left = tensors[node["left"]]
                right = tensors[node["right"]]
                rho = jnp.tensordot(left, right, axes=0)
                rho = _apply_block_sequence(
                    rho,
                    blocks,
                    blocks_h,
                    node["block_positions"],
                    node["block_apply_specs"],
                    node["block_param_indices"],
                )
                tensors[i] = rho
                continue

            if kind == "pre_green":
                left = tensors[node["left"]]
                right = tensors[node["right"]]
                rho = jnp.tensordot(left, right, axes=0)
                rho = _trace_tensor(rho, node["trace_pairs"])
                rho = jnp.transpose(rho, node["perm"])
                rho = _apply_block_sequence(
                    rho,
                    blocks,
                    blocks_h,
                    node["block_positions"],
                    node["block_apply_specs"],
                    node["block_param_indices"],
                )
                tensors[i] = rho
                continue

            # kind == "post_green"
            rho = tensors[node["main"]]
            rho = _trace_tensor(rho, node["trace_pairs"])
            rho = jnp.transpose(rho, node["perm"])
            rho = _apply_block_sequence(
                rho,
                blocks,
                blocks_h,
                node["block_positions"],
                node["block_apply_specs"],
                node["block_param_indices"],
            )
            tensors[i] = rho

        z = PaulyZ()
        expvals = []
        for out in outputs:
            rho = tensors[out["root"]]
            rho = _trace_tensor(rho, out["trace_pairs"])
            expvals.append(jnp.real(jnp.einsum(z, [0, 1], rho, [0, 1], [])))
        return jnp.stack(expvals, axis=0)

    return _forward_impl


def _make_forward_with_cache_impl(meta: Dict[str, Any]):
    plan = meta["contraction_plan"]
    nodes = plan["nodes"]
    outputs = plan["outputs"]
    bmera_type = int(meta["type"])

    def _forward_with_cache(params: jnp.ndarray, x: jnp.ndarray):
        x = jnp.asarray(x)
        params_flat = _pack_params(params)
        blocks, blocks_h = _build_block_tensors(bmera_type, params_flat)

        tensors: List[Any] = [None] * len(nodes)
        for i, node in enumerate(nodes):
            kind = node["kind"]
            if kind == "empty":
                tensors[i] = jnp.asarray(1.0, dtype=blocks.dtype)
                continue

            if kind == "layer0":
                tensors[i] = _product_state_density(x, node["qubits"])
                continue

            if kind == "layer1":
                left = tensors[node["left"]]
                right = tensors[node["right"]]
                rho = jnp.tensordot(left, right, axes=0)
                tensors[i] = _apply_block_sequence(
                    rho,
                    blocks,
                    blocks_h,
                    node["block_positions"],
                    node["block_apply_specs"],
                    node["block_param_indices"],
                )
                continue

            if kind == "pre_green":
                left = tensors[node["left"]]
                right = tensors[node["right"]]
                rho = jnp.tensordot(left, right, axes=0)
                rho = _trace_tensor(rho, node["trace_pairs"])
                rho = jnp.transpose(rho, node["perm"])
                tensors[i] = _apply_block_sequence(
                    rho,
                    blocks,
                    blocks_h,
                    node["block_positions"],
                    node["block_apply_specs"],
                    node["block_param_indices"],
                )
                continue

            rho = tensors[node["main"]]
            rho = _trace_tensor(rho, node["trace_pairs"])
            rho = jnp.transpose(rho, node["perm"])
            tensors[i] = _apply_block_sequence(
                rho,
                blocks,
                blocks_h,
                node["block_positions"],
                node["block_apply_specs"],
                node["block_param_indices"],
            )

        z = PaulyZ()
        expvals = []
        for out in outputs:
            rho = tensors[out["root"]]
            rho = _trace_tensor(rho, out["trace_pairs"])
            expvals.append(jnp.real(jnp.einsum(z, [0, 1], rho, [0, 1], [])))

        cache = {
            "node_outputs": tuple(tensors),
            "blocks": blocks,
        }
        return jnp.stack(expvals, axis=0), cache

    return _forward_with_cache


def _get_forward_fns(n_qubits: int, type: int):
    key = (int(n_qubits), int(type))
    if key in _FORWARD_FNS_CACHE:
        return _FORWARD_FNS_CACHE[key]

    meta = _get_meta(n_qubits, type)
    raw_forward = _make_forward_impl(meta)
    forward_single_jit = jax.jit(raw_forward)
    forward_batch_jit = jax.jit(jax.vmap(raw_forward, in_axes=(None, 0)))
    _FORWARD_FNS_CACHE[key] = (forward_single_jit, forward_batch_jit)
    return _FORWARD_FNS_CACHE[key]


def _get_forward_with_cache_fns(n_qubits: int, type: int):
    key = (int(n_qubits), int(type))
    if key in _FORWARD_WITH_CACHE_FNS_CACHE:
        return _FORWARD_WITH_CACHE_FNS_CACHE[key]

    meta = _get_meta(n_qubits, type)
    raw_forward = _make_forward_with_cache_impl(meta)
    forward_single_jit = jax.jit(raw_forward)
    forward_batch_jit = jax.jit(jax.vmap(raw_forward, in_axes=(None, 0)))
    _FORWARD_WITH_CACHE_FNS_CACHE[key] = (forward_single_jit, forward_batch_jit)
    return _FORWARD_WITH_CACHE_FNS_CACHE[key]


def forward_single(params: jnp.ndarray, x: jnp.ndarray, meta: Dict[str, Any]) -> jnp.ndarray:
    params, x = _forward_inputs(params, x, meta, batched=False)
    fn_single, _ = _get_forward_fns(meta["n_qubits"], meta["type"])
    return fn_single(params, x)


def forward_batch(params: jnp.ndarray, X: jnp.ndarray, meta: Dict[str, Any]) -> jnp.ndarray:
    params, X = _forward_inputs(params, X, meta, batched=True)
    _, fn_batch = _get_forward_fns(meta["n_qubits"], meta["type"])
    return fn_batch(params, X)


def forward_single_with_cache(params: jnp.ndarray, x: jnp.ndarray, meta: Dict[str, Any]):
    params, x = _forward_inputs(params, x, meta, batched=False)
    fn_single, _ = _get_forward_with_cache_fns(meta["n_qubits"], meta["type"])
    return fn_single(params, x)


def forward_batch_with_cache(params: jnp.ndarray, X: jnp.ndarray, meta: Dict[str, Any]):
    params, X = _forward_inputs(params, X, meta, batched=True)
    _, fn_batch = _get_forward_with_cache_fns(meta["n_qubits"], meta["type"])
    return fn_batch(params, X)


def _cache_nbytes(cache: Dict[str, Any]) -> int:
    nbytes = 0
    for t in cache["node_outputs"]:
        nbytes += int(t.size * t.dtype.itemsize)
    nbytes += int(cache["blocks"].size * cache["blocks"].dtype.itemsize)
    return nbytes


def cache_profile(cache: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    nodes = meta["contraction_plan"]["nodes"]
    node_rows = []
    total_node_bytes = 0
    for i, t in enumerate(cache["node_outputs"]):
        nb = int(t.size * t.dtype.itemsize)
        total_node_bytes += nb
        node_rows.append(
            {
                "node_id": int(nodes[i]["node_id"]),
                "kind": nodes[i]["kind"],
                "support_qubits": tuple(nodes[i]["support_qubits"]),
                "tensor_ndim": int(t.ndim),
                "tensor_shape": tuple(int(s) for s in t.shape),
                "tensor_nbytes": nb,
            }
        )

    blocks = cache["blocks"]
    block_bytes = int(blocks.size * blocks.dtype.itemsize)
    return {
        "total_cache_nbytes": int(total_node_bytes + block_bytes),
        "node_tensors_nbytes": int(total_node_bytes),
        "block_tensors_nbytes": int(block_bytes),
        "rows": node_rows,
    }


class BMERANetwork:
    """BMERA tensor-network simulator (JAX-jitted by default)."""

    def __init__(self, n_qubits, type=1, params: np.ndarray = None, backend: str = "tensor"):
        if backend not in ("tensor", "auto", "legacy", "fast"):
            raise ValueError("backend must be one of 'tensor', 'auto', 'legacy', 'fast'")
        self.backend = "tensor"
        self.meta = _get_meta(n_qubits, type)
        self.n_qubits = self.meta["n_qubits"]
        self.type = self.meta["type"]
        n_per_block = 4 if self.type == 1 else 5
        self.depth = int(self.meta["depth"])
        self.width = self.n_qubits // 2
        self.params = (
            np.random.uniform(-np.pi, np.pi, (self.depth * self.n_qubits, n_per_block))
            if params is None
            else params
        )

    def compute_expval(self, in_data: jnp.ndarray, qubit: int):
        qubit = _validate_qubit(qubit, self.n_qubits)
        return forward_single(self.params, in_data, self.meta)[qubit]

    def compute_all_expval(self, in_data: jnp.ndarray):
        return jnp.real(forward_single(jnp.asarray(self.params), jnp.asarray(in_data), self.meta))

    def compute_rho(self, in_data: np.array, layer: int, qubits: np.array, after_green: bool):
        raise NotImplementedError("compute_rho is not exposed in the compiled tensor-network backend")


def compute_expval(type: int, params: np.ndarray, in_data: np.ndarray, n_qubits: int, qubit: int):
    meta = _get_meta(n_qubits, type)
    qubit = _validate_qubit(qubit, meta["n_qubits"])
    return forward_single(params, in_data, meta)[qubit]


def compute_all_expval(type: int, params: jnp.ndarray, in_data: jnp.ndarray, n_qubits: int):
    meta = _get_meta(n_qubits, type)
    return jnp.real(forward_single(params, in_data, meta))


def model_output(type: int, params: jnp.ndarray, weights: jnp.ndarray, in_data: jnp.ndarray, n_qubits: int):
    return jnp.real(jnp.dot(weights, compute_all_expval(type, params, in_data, n_qubits)))


def compute_expval_real(type: int, params: jnp.ndarray, in_data: jnp.ndarray, n_qubits: int, qubit: int):
    return jnp.real(compute_expval(type, params, in_data, n_qubits, qubit))


def weights(a: int):
    return jnp.array(np.random.uniform(0, 1, 2**a))


compute_grad = jax.grad(compute_expval_real, 1)


compute_expval_and_grad = jax.value_and_grad(compute_expval_real, 1)


compute_model_grad = jax.grad(model_output, (1, 2))
