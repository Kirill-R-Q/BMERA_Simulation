"""BMERA qubit permutations and contraction plans."""

from __future__ import annotations

from operator import index
from typing import Any, Dict, List, Tuple
import numpy as np


def _legacy_swap_indices(indices: np.array, layer: int, after_green: bool) -> np.array:
    if after_green:
        blocks = indices // (2 ** (layer + 1))
        indices = indices % (2 ** (layer + 1))
        res = indices // (2**layer)
        return (2 * indices + 2 - res) % (2 ** (layer + 1)) + blocks * (2 ** (layer + 1))
    blocks = indices // (2**layer)
    indices = indices % (2**layer)
    return -1 * (-1 * (indices - 1)) % (2**layer) + blocks * (2**layer)


def _legacy_get_main_indices(indices: np.array, layer: int) -> np.array:
    blocks = indices // (2**layer)
    indices = (indices + 1) % (2**layer)
    return indices + blocks * (2**layer)


def _legacy_get_trace_indices(indices: np.array, layer: int) -> np.array:
    main_indices = _legacy_get_main_indices(indices, layer)
    qubit_groups = np.unique(main_indices // 2)
    return np.setdiff1d(np.concatenate((2 * qubit_groups, 2 * qubit_groups + 1)), main_indices)


def _legacy_get_indices(indices: np.array, layer: int) -> np.array:
    main_indices = _legacy_get_main_indices(indices, layer)
    qubit_groups = np.unique(main_indices // 2)
    return np.sort(np.concatenate((2 * qubit_groups, 2 * qubit_groups + 1)))


def _legacy_get_left_main_indices(indices: np.array, layer: int):
    qubit_groups = indices // 2
    block_size = 2**layer
    left_main_qubits = (qubit_groups - 1) % (block_size // 2) + ((qubit_groups) // (block_size // 2)) * block_size
    return np.unique(left_main_qubits)


def _legacy_get_right_main_indices(indices: np.array, layer: int):
    qubit_groups = indices // 2
    block_size = 2**layer
    right_main_qubits = (
        (qubit_groups + block_size // 2) % (block_size // 2)
        + (block_size // 2)
        + ((qubit_groups) // (block_size // 2)) * block_size
    )
    return np.unique(right_main_qubits)


def _legacy_get_left_trace_indices(indices: np.array, layer: int):
    left_main_indices = _legacy_get_left_main_indices(indices, layer)
    qubit_groups = np.unique(left_main_indices // 2)
    return np.setdiff1d(np.concatenate((2 * qubit_groups, 2 * qubit_groups + 1)), left_main_indices)


def _legacy_get_right_trace_indices(indices: np.array, layer: int):
    right_main_indices = _legacy_get_right_main_indices(indices, layer)
    qubit_groups = np.unique(right_main_indices // 2)
    return np.setdiff1d(np.concatenate((2 * qubit_groups, 2 * qubit_groups + 1)), right_main_indices)


def _legacy_get_left_indices(indices: np.array, layer: int):
    left_main_indices = _legacy_get_left_main_indices(indices, layer)
    qubit_groups = np.unique(left_main_indices // 2)
    return np.sort(np.concatenate((2 * qubit_groups, 2 * qubit_groups + 1)))


def _legacy_get_right_indices(indices: np.array, layer: int):
    right_main_indices = _legacy_get_right_main_indices(indices, layer)
    qubit_groups = np.unique(right_main_indices // 2)
    return np.sort(np.concatenate((2 * qubit_groups, 2 * qubit_groups + 1)))


_META_CACHE: Dict[tuple[int, int], Dict[str, Any]] = {}


def _trace_pairs_and_remaining(qubits: Tuple[int, ...], trace_qubits: np.ndarray) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]:
    trace_qubits = np.unique(trace_qubits.astype(np.int32))
    remaining = list(qubits)
    axes_info = []
    for q in trace_qubits:
        pos = remaining.index(int(q))
        axes_info.append((2 * pos, 2 * pos + 1, int(q)))
    axes_info.sort(key=lambda t: t[0], reverse=True)
    for _, _, q in axes_info:
        remaining.remove(q)
    return tuple((a, b) for a, b, _ in axes_info), tuple(remaining)


def _reorder_perm_and_qubits(current_qubits: Tuple[int, ...], new_qubit_order: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if len(current_qubits) != int(new_qubit_order.size):
        raise ValueError("invalid reorder request")
    pairs = [(int(new_qubit_order[i]), i) for i in range(len(current_qubits))]
    pairs.sort(key=lambda t: t[0])
    perm = []
    for _, old_pos in pairs:
        perm.extend([2 * old_pos, 2 * old_pos + 1])
    return tuple(perm), tuple(sorted(int(v) for v in new_qubit_order))


def _block_positions_and_params(
    out_qubits: Tuple[int, ...],
    qubit_groups: np.ndarray,
    width: int,
    layer_index: int,
    color: str,
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]:
    positions = []
    param_indices = []
    for q in np.unique(qubit_groups.astype(np.int32)):
        q0 = 2 * int(q)
        q1 = q0 + 1
        p0 = out_qubits.index(q0)
        p1 = out_qubits.index(q1)
        positions.append((p0, p1))
        base = 2 * (layer_index * width + int(q))
        param_indices.append(base if color == "red" else base + 1)
    return tuple(positions), tuple(param_indices)


def _unitary_apply_spec(n_local: int, pos0: int, pos1: int) -> Dict[str, Tuple[int, ...]]:
    in0 = 2 * int(pos0)
    out0 = in0 + 1
    in1 = 2 * int(pos1)
    out1 = in1 + 1
    target_axes = (in0, in1, out0, out1)
    perm = tuple(axis for axis in range(2 * int(n_local)) if axis not in target_axes) + target_axes
    inv_perm = tuple(int(idx) for idx in np.argsort(np.asarray(perm, dtype=np.int32)))
    return {
        "perm": perm,
        "inv_perm": inv_perm,
    }


def _build_contraction_plan(n_qubits: int) -> Dict[str, Any]:
    depth = int(np.log2(n_qubits))
    width = n_qubits // 2
    nodes: List[Dict[str, Any]] = []
    memo: Dict[Tuple[int, Tuple[int, ...], bool], int] = {}

    def register(layer: int, qubits: np.ndarray, after_green: bool) -> int:
        qubits = np.unique(np.asarray(qubits, dtype=np.int32))
        key = (int(layer), tuple(int(v) for v in qubits), bool(after_green))
        if key in memo:
            return memo[key]

        if qubits.size == 0:
            node = {
                "kind": "empty",
                "layer": int(layer),
                "after_green": bool(after_green),
                "out_qubits": tuple(),
            }
        elif layer == 0:
            node = {
                "kind": "layer0",
                "layer": 0,
                "after_green": bool(after_green),
                "qubits": tuple(int(v) for v in qubits),
                "out_qubits": tuple(int(v) for v in qubits),
            }
        elif layer == 1:
            if qubits.size != 2:
                raise ValueError("invalid layer-1 contraction request")
            left_id = register(0, np.array([qubits[0]], dtype=np.int32), False)
            right_id = register(0, np.array([qubits[1]], dtype=np.int32), False)
            q_group = int(qubits[0] // 2)
            out_qubits = tuple(int(v) for v in qubits)
            positions, pidx = _block_positions_and_params(
                out_qubits=out_qubits,
                qubit_groups=np.array([q_group], dtype=np.int32),
                width=width,
                layer_index=0,
                color="green",
            )
            node = {
                "kind": "layer1",
                "layer": 1,
                "after_green": True,
                "left": left_id,
                "right": right_id,
                "block_positions": positions,
                "block_param_indices": pidx,
                "out_qubits": out_qubits,
            }
        elif not after_green:
            left_trace = _legacy_get_left_trace_indices(qubits, layer)
            right_trace = _legacy_get_right_trace_indices(qubits, layer)
            left_qubits = _legacy_get_left_indices(qubits, layer)
            right_qubits = _legacy_get_right_indices(qubits, layer)

            left_id = register(layer - 1, left_qubits, True)
            right_id = register(layer - 1, right_qubits, True)

            left_out = tuple(nodes[left_id]["out_qubits"])
            right_out = tuple(nodes[right_id]["out_qubits"])
            concat_out = left_out + right_out

            trace_pairs, rem_qubits = _trace_pairs_and_remaining(concat_out, np.concatenate((left_trace, right_trace)))
            swapped = _legacy_swap_indices(np.array(rem_qubits, dtype=np.int32), layer - 1, True)
            perm, out_qubits = _reorder_perm_and_qubits(rem_qubits, swapped)

            qubit_groups = np.unique(qubits // 2)
            positions, pidx = _block_positions_and_params(
                out_qubits=out_qubits,
                qubit_groups=qubit_groups,
                width=width,
                layer_index=layer - 1,
                color="red",
            )
            node = {
                "kind": "pre_green",
                "layer": int(layer),
                "after_green": False,
                "left": left_id,
                "right": right_id,
                "trace_pairs": trace_pairs,
                "perm": perm,
                "block_positions": positions,
                "block_param_indices": pidx,
                "out_qubits": out_qubits,
            }
        else:
            trace_qubits = _legacy_get_trace_indices(qubits, layer)
            all_qubits = _legacy_get_indices(qubits, layer)
            main_id = register(layer, all_qubits, False)

            main_out = tuple(nodes[main_id]["out_qubits"])
            trace_pairs, rem_qubits = _trace_pairs_and_remaining(main_out, trace_qubits)
            swapped = _legacy_swap_indices(np.array(rem_qubits, dtype=np.int32), layer, False)
            perm, out_qubits = _reorder_perm_and_qubits(rem_qubits, swapped)

            qubit_groups = np.unique(qubits // 2)
            positions, pidx = _block_positions_and_params(
                out_qubits=out_qubits,
                qubit_groups=qubit_groups,
                width=width,
                layer_index=layer - 1,
                color="green",
            )
            node = {
                "kind": "post_green",
                "layer": int(layer),
                "after_green": True,
                "main": main_id,
                "trace_pairs": trace_pairs,
                "perm": perm,
                "block_positions": positions,
                "block_param_indices": pidx,
                "out_qubits": out_qubits,
            }

        node_id = len(nodes)
        nodes.append(node)
        memo[key] = node_id
        return node_id

    outputs = []
    for q in range(n_qubits):
        q_group = q // 2
        root_qubits = np.array([2 * q_group, 2 * q_group + 1], dtype=np.int32)
        root_id = register(depth, root_qubits, True)
        trace_qubit = 2 * q_group + 1 if (2 * q_group) == q else 2 * q_group
        root_out = tuple(nodes[root_id]["out_qubits"])
        final_trace_pairs, _ = _trace_pairs_and_remaining(root_out, np.array([trace_qubit], dtype=np.int32))
        outputs.append(
            {
                "qubit": int(q),
                "root": int(root_id),
                "trace_pairs": final_trace_pairs,
            }
        )

    for node_id, node in enumerate(nodes):
        kind = node["kind"]
        if kind == "layer1":
            child_ids = (int(node["left"]), int(node["right"]))
        elif kind == "pre_green":
            child_ids = (int(node["left"]), int(node["right"]))
        elif kind == "post_green":
            child_ids = (int(node["main"]),)
        else:
            child_ids = tuple()
        node["node_id"] = int(node_id)
        node["child_ids"] = child_ids
        node["support_qubits"] = tuple(int(q) for q in node["out_qubits"])
        if "block_positions" in node:
            node["block_apply_specs"] = tuple(
                _unitary_apply_spec(len(node["out_qubits"]), int(p0), int(p1))
                for p0, p1 in node["block_positions"]
            )
        else:
            node["block_apply_specs"] = tuple()

    return {
        "depth": int(depth),
        "width": int(width),
        "nodes": nodes,
        "outputs": outputs,
    }


def _validate_topology(n_qubits, block_type):
    try:
        n_qubits, block_type = index(n_qubits), index(block_type)
    except TypeError as exc:
        raise ValueError("number of qubits and block type must be integers") from exc
    if n_qubits < 2 or n_qubits & (n_qubits - 1):
        raise ValueError("number of qubits should be a power of 2 and >= 2")
    if block_type not in (1, 2, 3):
        raise ValueError("type can be only 1 to 3")
    return n_qubits, block_type


def build_bmera_meta(n_qubits: int, type: int) -> Dict[str, Any]:
    """Build the contraction plan and parameter layout for a BMERA circuit."""
    n_qubits, type = _validate_topology(n_qubits, type)
    plan = _build_contraction_plan(n_qubits)
    n_per_block = 4 if type == 1 else 5
    return {
        "n_qubits": int(n_qubits),
        "depth": int(plan["depth"]),
        "width": int(plan["width"]),
        "type": int(type),
        "n_per_block": int(n_per_block),
        "contraction_plan": plan,
    }


def _get_meta(n_qubits: int, type: int) -> Dict[str, Any]:
    key = _validate_topology(n_qubits, type)
    if key not in _META_CACHE:
        _META_CACHE[key] = build_bmera_meta(*key)
    return _META_CACHE[key]


def active_block_indices(meta: Dict[str, Any]) -> tuple[int, ...]:
    """Return the parameter rows used by the contraction plan."""
    return tuple(sorted({
        index
        for node in meta["contraction_plan"]["nodes"]
        for index in node.get("block_param_indices", ())
    }))
