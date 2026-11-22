"""Optional TensorCircuit reference implementation."""

from __future__ import annotations

from threading import RLock
from importlib.metadata import version
from typing import Any, Dict, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from ..benchmarking import _block_tree_until_ready, benchmark_callable, benchmark_context
from ..gradients import _loss_and_grad_autodiff_full
from ..simulator import _forward_inputs, _pack_params, _supervised_inputs, forward_batch
from ..topology import _get_meta


_TENSORCIRCUIT_CONFIG_LOCK = RLock()
_TENSORCIRCUIT_FNS_CACHE: Dict[tuple[int, int], Tuple[Any, Any]] = {}


_TENSORCIRCUIT_GRAD_FNS_CACHE: Dict[tuple[int, int], Any] = {}


def _import_tensorcircuit():
    try:
        import tensorcircuit as tc
    except ImportError as exc:
        raise ImportError("Install the `reference` extra to use TensorCircuit-NG.") from exc
    except TypeError as exc:
        if "newshape" not in str(exc):
            raise
        raise ImportError(
            "This NumPy version requires TensorCircuit-NG. Install the `reference` extra "
            "in an environment without the legacy tensorcircuit package."
        ) from exc
    return tc


def _tc_bmera_swap_layer(circuit: Any, n_total: int, n: int, start_qubit: int = 0) -> None:
    target_map = {}
    for i in range(int(n_total)):
        if start_qubit <= i < n - 1 + start_qubit:
            target_map[i] = 2 * (i + 1) - start_qubit
        elif i == n - 1 + start_qubit:
            target_map[i] = start_qubit
        elif 2 * n - 1 + start_qubit >= i > n - 1 + start_qubit:
            target_map[i] = 2 * (i - n) + 1 - start_qubit
        else:
            target_map[i] = i

    current_pos = list(range(int(n_total)))
    physical_pos_to_logical = list(range(int(n_total)))

    for physical_wire in range(int(n_total)):
        logical_qubit_that_should_be_here = -1
        for k, v in target_map.items():
            if v == physical_wire:
                logical_qubit_that_should_be_here = k
                break

        current_physical_pos_of_target = current_pos[logical_qubit_that_should_be_here]
        if current_physical_pos_of_target == physical_wire:
            continue

        circuit.swap(physical_wire, current_physical_pos_of_target)
        logical_qubit_that_was_here = physical_pos_to_logical[physical_wire]
        current_pos[logical_qubit_that_should_be_here] = physical_wire
        current_pos[logical_qubit_that_was_here] = current_physical_pos_of_target
        physical_pos_to_logical[physical_wire] = logical_qubit_that_should_be_here
        physical_pos_to_logical[current_physical_pos_of_target] = logical_qubit_that_was_here


def _tc_apply_bmera_block(circuit: Any, bmera_type: int, params: jnp.ndarray, q_top: int, q_bottom: int) -> None:
    circuit.rx(q_top, theta=params[0])
    circuit.rx(q_bottom, theta=params[1])
    circuit.rz(q_top, theta=params[2])
    circuit.rz(q_bottom, theta=params[3])

    if int(bmera_type) == 1:
        circuit.cnot(q_bottom, q_top)
    elif int(bmera_type) == 2:
        circuit.crx(q_bottom, q_top, theta=params[4])
    elif int(bmera_type) == 3:
        circuit.crz(q_bottom, q_top, theta=params[4])
    else:
        raise ValueError("type can be only 1 to 3")


def _make_tensorcircuit_forward_impl(meta: Dict[str, Any]):
    tc = _import_tensorcircuit()
    n_qubits = int(meta["n_qubits"])
    depth = int(meta["depth"])
    width = int(meta["width"])
    bmera_type = int(meta["type"])

    def _forward_impl(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        params_flat = _pack_params(params)
        network_params = params_flat.reshape((depth, width * 2, int(meta["n_per_block"])))
        with _TENSORCIRCUIT_CONFIG_LOCK, tc.runtime_backend("jax"), tc.runtime_dtype("complex64"):
            circuit = tc.Circuit(n_qubits)

            for q in range(n_qubits):
                circuit.ry(q, theta=2.0 * x[q])

            for j in range(1, depth + 1):
                for start_qubit in range(0, n_qubits, 2**j):
                    _tc_bmera_swap_layer(circuit, n_qubits, 2 ** (j - 1), start_qubit)

                layer_params = network_params[j - 1]
                if j == 1:
                    for start_qubit in range(0, n_qubits, 2):
                        _tc_apply_bmera_block(
                            circuit,
                            bmera_type,
                            layer_params[start_qubit + 1],
                            start_qubit,
                            start_qubit + 1,
                        )
                else:
                    for start_qubit in range(0, n_qubits, 2**j):
                        for i in range(2**j // 2):
                            _tc_apply_bmera_block(
                                circuit,
                                bmera_type,
                                layer_params[2 * i + start_qubit],
                                2 * i + start_qubit,
                                2 * i + 1 + start_qubit,
                            )
                        for i in range(2**j - 1):
                            circuit.swap(i + start_qubit, i + 1 + start_qubit)
                        for i in range(2**j // 2):
                            _tc_apply_bmera_block(
                                circuit,
                                bmera_type,
                                layer_params[start_qubit + 2 * i + 1],
                                2 * i + start_qubit,
                                2 * i + 1 + start_qubit,
                            )

            return jnp.real(jnp.stack([circuit.expectation_ps(z=[q]) for q in range(n_qubits)], axis=0))

    return _forward_impl


def _get_tensorcircuit_forward_fns(n_qubits: int, type: int):
    key = (int(n_qubits), int(type))
    if key in _TENSORCIRCUIT_FNS_CACHE:
        return _TENSORCIRCUIT_FNS_CACHE[key]

    meta = _get_meta(n_qubits, type)
    raw_forward = _make_tensorcircuit_forward_impl(meta)
    forward_single_jit = jax.jit(raw_forward)
    forward_batch_jit = jax.jit(jax.vmap(raw_forward, in_axes=(None, 0)))
    _TENSORCIRCUIT_FNS_CACHE[key] = (forward_single_jit, forward_batch_jit)
    return _TENSORCIRCUIT_FNS_CACHE[key]


def tensorcircuit_forward_single(params: jnp.ndarray, x: jnp.ndarray, meta: Dict[str, Any]) -> jnp.ndarray:
    params, x = _forward_inputs(params, x, meta, batched=False)
    fn_single, _ = _get_tensorcircuit_forward_fns(meta["n_qubits"], meta["type"])
    return fn_single(params, x)


def tensorcircuit_forward_batch(params: jnp.ndarray, X: jnp.ndarray, meta: Dict[str, Any]) -> jnp.ndarray:
    params, X = _forward_inputs(params, X, meta, batched=True)
    _, fn_batch = _get_tensorcircuit_forward_fns(meta["n_qubits"], meta["type"])
    return fn_batch(params, X)


def tensorcircuit_loss(
    params: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    meta: Dict[str, Any],
    head_weights: jnp.ndarray,
) -> jnp.ndarray:
    params, X, y, head_weights = _supervised_inputs(params, X, y, meta, head_weights)
    preds_all = tensorcircuit_forward_batch(params, X, meta)
    preds = preds_all @ head_weights
    err = preds - y
    return jnp.mean(err * err)


def _get_tensorcircuit_grad_fn(n_qubits: int, type: int):
    key = (int(n_qubits), int(type))
    if key not in _TENSORCIRCUIT_GRAD_FNS_CACHE:
        meta = _get_meta(n_qubits, type)
        _TENSORCIRCUIT_GRAD_FNS_CACHE[key] = jax.jit(
            lambda p, X, y, h: jax.value_and_grad(tensorcircuit_loss, (0, 4))(p, X, y, meta, h)
        )
    return _TENSORCIRCUIT_GRAD_FNS_CACHE[key]


def loss_and_grad_tensorcircuit(
    params: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    meta: Dict[str, Any],
    head_weights: jnp.ndarray,
):
    params, X, y, head_weights = _supervised_inputs(params, X, y, meta, head_weights)
    grad_fn = _get_tensorcircuit_grad_fn(meta["n_qubits"], meta["type"])
    lval, (grad_params, grad_head) = grad_fn(params, X, y, head_weights)
    return lval, grad_params, grad_head


def validate_tensorcircuit_backend(
    n_qubits_list: Tuple[int, ...] = (4, 8),
    types: Tuple[int, ...] = (1, 2, 3),
    batch_size: int = 2,
    dtype=jnp.float32,
    seed: int = 123,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for bmera_type in types:
        for n_qubits in n_qubits_list:
            meta = _get_meta(int(n_qubits), int(bmera_type))
            param_shape = (int(meta["depth"]) * int(n_qubits), int(meta["n_per_block"]))
            params = jnp.asarray(rng.uniform(-0.4, 0.4, size=param_shape), dtype=dtype)
            X = jnp.asarray(rng.uniform(-0.5, 0.5, size=(int(batch_size), int(n_qubits))), dtype=dtype)
            y = jnp.asarray(rng.uniform(-1.0, 1.0, size=(int(batch_size),)), dtype=dtype)
            head = jnp.asarray(rng.uniform(-0.4, 0.4, size=(int(n_qubits),)), dtype=dtype)

            tn_forward = forward_batch(params, X, meta)
            tc_forward = tensorcircuit_forward_batch(params, X, meta)
            tn_loss, tn_grad_p, tn_grad_h = _loss_and_grad_autodiff_full(params, X, y, meta, head)
            tc_loss, tc_grad_p, tc_grad_h = loss_and_grad_tensorcircuit(params, X, y, meta, head)
            _block_tree_until_ready((tn_forward, tc_forward, tn_loss, tn_grad_p, tn_grad_h, tc_loss, tc_grad_p, tc_grad_h))

            rows.append(
                {
                    "type": int(bmera_type),
                    "n_qubits": int(n_qubits),
                    "batch_size": int(batch_size),
                    "forward_max_abs_error": float(jnp.max(jnp.abs(tn_forward - tc_forward))),
                    "loss_abs_error": float(jnp.abs(tn_loss - tc_loss)),
                    "grad_params_max_abs_error": float(jnp.max(jnp.abs(tn_grad_p - tc_grad_p))),
                    "grad_head_max_abs_error": float(jnp.max(jnp.abs(tn_grad_h - tc_grad_h))),
                    "grad_max_abs_error": float(
                        jnp.maximum(jnp.max(jnp.abs(tn_grad_p - tc_grad_p)), jnp.max(jnp.abs(tn_grad_h - tc_grad_h)))
                    ),
                }
            )
    return rows


def benchmark_tensorcircuit_backend_scaling(
    n_qubits_list: Tuple[int, ...] = (8, 16, 32),
    batch_sizes: Tuple[int, ...] = (1, 4, 16),
    types: Tuple[int, ...] = (1,),
    repeats: int = 3,
    warmup: int = 1,
    dtype=jnp.float32,
    seed: int = 321,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for bmera_type in types:
        for n_qubits in n_qubits_list:
            meta = _get_meta(int(n_qubits), int(bmera_type))
            param_shape = (int(meta["depth"]) * int(n_qubits), int(meta["n_per_block"]))
            for batch_size in batch_sizes:
                params = jnp.asarray(rng.uniform(-0.4, 0.4, size=param_shape), dtype=dtype)
                X = jnp.asarray(rng.uniform(-0.5, 0.5, size=(int(batch_size), int(n_qubits))), dtype=dtype)
                y = jnp.asarray(rng.uniform(-1.0, 1.0, size=(int(batch_size),)), dtype=dtype)
                head = jnp.asarray(rng.uniform(-0.4, 0.4, size=(int(n_qubits),)), dtype=dtype)

                tn_forward_bench = benchmark_callable(lambda: forward_batch(params, X, meta), repeats, warmup)
                tc_forward_bench = benchmark_callable(lambda: tensorcircuit_forward_batch(params, X, meta), repeats, warmup)
                tn_grad_bench = benchmark_callable(
                    lambda: _loss_and_grad_autodiff_full(params, X, y, meta, head), repeats, warmup
                )
                tc_grad_bench = benchmark_callable(
                    lambda: loss_and_grad_tensorcircuit(params, X, y, meta, head), repeats, warmup
                )

                rows.append(
                    {
                        "type": int(bmera_type),
                        "n_qubits": int(n_qubits),
                        "batch_size": int(batch_size),
                        **benchmark_context(params.device, dtype, repeats, warmup, seed),
                        "tensorcircuit_version": version("tensorcircuit-ng"),
                        "existing_forward_compile_plus_first_ms": tn_forward_bench["compile_plus_first_ms"],
                        "existing_forward_steady_ms": tn_forward_bench["steady_mean_ms"],
                        "existing_forward_steady_std_ms": tn_forward_bench["steady_std_ms"],
                        "tensorcircuit_forward_compile_plus_first_ms": tc_forward_bench["compile_plus_first_ms"],
                        "tensorcircuit_forward_steady_ms": tc_forward_bench["steady_mean_ms"],
                        "tensorcircuit_forward_steady_std_ms": tc_forward_bench["steady_std_ms"],
                        "existing_grad_compile_plus_first_ms": tn_grad_bench["compile_plus_first_ms"],
                        "existing_grad_steady_ms": tn_grad_bench["steady_mean_ms"],
                        "existing_grad_steady_std_ms": tn_grad_bench["steady_std_ms"],
                        "tensorcircuit_grad_compile_plus_first_ms": tc_grad_bench["compile_plus_first_ms"],
                        "tensorcircuit_grad_steady_ms": tc_grad_bench["steady_mean_ms"],
                        "tensorcircuit_grad_steady_std_ms": tc_grad_bench["steady_std_ms"],
                    }
                )
    return rows
