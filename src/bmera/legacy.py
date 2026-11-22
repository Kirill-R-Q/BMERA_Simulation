"""Recursive implementation and density matrix helpers for legacy callers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from .gates import (
    H,
    InitState,
    PaulyZ,
    createBMERABlock,
)
from .topology import (
    _legacy_get_indices,
    _legacy_get_left_indices,
    _legacy_get_left_trace_indices,
    _legacy_get_right_indices,
    _legacy_get_right_trace_indices,
    _legacy_get_trace_indices,
    _legacy_swap_indices,
)


class density_matrix:
    def __init__(self, tensor: np.array, qubits: np.array):
        self.qubits = np.asarray(qubits, dtype=np.int32)
        num_qubits = self.qubits.size
        expected_num_dims = 2 * num_qubits
        if len(tensor.shape) != expected_num_dims:
            raise ValueError(
                f"Tensor has {len(tensor.shape)} dimensions, expected {expected_num_dims} for {num_qubits} qubits"
            )
        if not all(s == 2 for s in tensor.shape):
            raise ValueError(f"All tensor dimensions must be 2. Got shape {tensor.shape}")
        self.tensor = tensor
        self.in_indices = 2 * self.qubits
        self.out_indices = 2 * self.qubits + 1
        self.indices = np.sort(np.concatenate((self.in_indices, self.out_indices)))

    def trace_rho(self, trace_qubits):
        trace_qubits = np.unique(trace_qubits)
        tensor_to_process = self.tensor
        remaining_logical_qubits = list(self.qubits)

        axes_to_trace_info = []
        for q_logical_to_trace in trace_qubits:
            pos_in_current_list = remaining_logical_qubits.index(q_logical_to_trace)
            axes_to_trace_info.append((2 * pos_in_current_list, 2 * pos_in_current_list + 1, q_logical_to_trace))

        axes_to_trace_info.sort(key=lambda x: x[0], reverse=True)

        for ax1, ax2, q_logical_traced in axes_to_trace_info:
            tensor_to_process = jnp.trace(tensor_to_process, axis1=ax1, axis2=ax2)
            remaining_logical_qubits.remove(q_logical_traced)

        new_qubits = np.array(remaining_logical_qubits, dtype=np.int32)
        return density_matrix(tensor_to_process, new_qubits)

    def reorder_qubits(self, new_qubit_order: np.array):
        num_qubits = self.qubits.size
        new_qubit_order = np.asarray(new_qubit_order)

        if new_qubit_order.size != num_qubits:
            raise ValueError(
                f"Size of new_qubit_order ({new_qubit_order.size}) must match number of qubits ({num_qubits})"
            )
        if np.unique(new_qubit_order).size != num_qubits:
            raise ValueError("new_qubit_order must contain unique qubit labels")

        final_qubits = np.sort(new_qubit_order)
        pairs = []
        for i in range(num_qubits):
            original_qubit = self.qubits[i]
            new_label = new_qubit_order[np.where(self.qubits == original_qubit)[0][0]]
            pairs.append((new_label, i))

        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        permutation_axes = []
        for _, original_pos in sorted_pairs:
            permutation_axes.extend([2 * original_pos, 2 * original_pos + 1])

        new_tensor = self.tensor.transpose(permutation_axes)
        return density_matrix(new_tensor, final_qubits)

    def get_diag(self):
        new_axis_order = np.concatenate((np.arange(0, self.tensor.ndim, 2), np.arange(1, self.tensor.ndim + 1, 2)))
        return np.diag(self.tensor.transpose(new_axis_order).reshape((2 ** self.qubits.size, 2 ** self.qubits.size)))


class _LegacyBMERANetwork:
    def __init__(self, n_qubits, type=1, params: np.array = None):
        if (n_qubits - np.exp2(int(np.log2(n_qubits))) >= 1):
            raise ValueError("number of qubits should be a power of 2")
        if n_qubits == 1:
            raise ValueError("number of qubits should be more than 1")

        self.depth = int(np.log2(n_qubits))
        self.width = n_qubits // 2
        self.n_qubits = n_qubits
        self.rho_map = {}

        if type == 1:
            n_per_block = 4
        elif type in (2, 3):
            n_per_block = 5
        else:
            raise ValueError("type can be only 1 to 3")
        self.type = type

        self.params = (
            params
            if params is not None
            else np.random.uniform(-np.pi, np.pi, (self.depth * self.width * 2, n_per_block))
        )
        self.params = self.params.reshape((self.depth * self.width * 2, n_per_block))

        self.red_blocks = []
        self.green_blocks = []
        for i in range(self.width * self.depth):
            self.red_blocks.append(createBMERABlock(type, self.params[2 * i]))
            self.green_blocks.append(createBMERABlock(type, self.params[2 * i + 1]))

        self.red_blocks = jnp.array(self.red_blocks).reshape((self.depth, self.width, 2, 2, 2, 2))
        self.green_blocks = jnp.array(self.green_blocks).reshape((self.depth, self.width, 2, 2, 2, 2))

    def compute_rho(self, in_data: np.array, layer: int, qubits: np.array, after_green: bool) -> density_matrix:
        cache_key = (layer, tuple(qubits), after_green)
        if self.rho_map.get(cache_key) is not None:
            return self.rho_map[cache_key]

        qubits = np.unique(qubits)
        qubit_groups = np.unique(qubits // 2)

        if qubits.size == 0:
            rho = density_matrix(np.array(1.0), np.array([]))
            self.rho_map[cache_key] = rho
            return rho

        if layer == 0:
            return density_matrix(
                jnp.tensordot(InitState(in_data[qubits]), InitState(in_data[qubits]), axes=0).reshape(2, 2),
                qubits,
            )

        if layer == 1:
            rho_left = self.compute_rho(in_data, 0, np.array([qubits[0]]), False)
            rho_right = self.compute_rho(in_data, 0, np.array([qubits[1]]), False)
            rho = density_matrix(jnp.tensordot(rho_left.tensor, rho_right.tensor, 0), qubits)
            U = self.green_blocks[0, qubit_groups[0]]
            U_H = H(U)
            new_rho_tensor = jnp.einsum(
                rho.tensor,
                [0, 1, 2, 3],
                U,
                [4, 0, 5, 2],
                U_H,
                [1, 6, 3, 7],
                [4, 6, 5, 7],
            )
            return density_matrix(new_rho_tensor, qubits)

        if not after_green:
            left_trace_qubits = _legacy_get_left_trace_indices(qubits, layer)
            right_trace_qubits = _legacy_get_right_trace_indices(qubits, layer)
            left_qubits = _legacy_get_left_indices(qubits, layer)
            right_qubits = _legacy_get_right_indices(qubits, layer)

            rho_left = self.compute_rho(in_data, layer - 1, left_qubits, True)
            rho_right = self.compute_rho(in_data, layer - 1, right_qubits, True)
            rho = density_matrix(
                jnp.tensordot(rho_left.tensor, rho_right.tensor, 0),
                np.concatenate((rho_left.qubits, rho_right.qubits)),
            )
            rho = rho.trace_rho(np.concatenate((left_trace_qubits, right_trace_qubits)))
            rho = rho.reorder_qubits(_legacy_swap_indices(rho.qubits, layer - 1, True))

            for q in qubit_groups:
                U = self.red_blocks[layer - 1, q]
                U_H = H(U)
                U_indices = np.array([4 + 4 * self.width, 4 * q, 6 + 4 * self.width, 2 + 4 * q])
                U_H_indices = np.array([1 + 4 * q, 5 + 4 * self.width, 3 + 4 * q, 7 + 4 * self.width])
                result_indices = np.concatenate(
                    (
                        rho.indices[rho.indices < 4 * q],
                        np.array([4 + 4 * self.width, 5 + 4 * self.width, 6 + 4 * self.width, 7 + 4 * self.width]),
                        rho.indices[rho.indices > 4 * q + 3],
                    )
                )
                rho.tensor = jnp.einsum(rho.tensor, rho.indices, U, U_indices, U_H, U_H_indices, result_indices)

            self.rho_map[cache_key] = rho
            return rho

        trace_qubits = _legacy_get_trace_indices(qubits, layer)
        all_qubits = _legacy_get_indices(qubits, layer)
        rho = self.compute_rho(in_data, layer, all_qubits, False)
        rho = rho.trace_rho(trace_qubits)
        rho = rho.reorder_qubits(_legacy_swap_indices(rho.qubits, layer, False))

        for q in qubit_groups:
            U = self.green_blocks[layer - 1, q]
            U_H = H(U)
            U_indices = np.array([4 + 4 * self.width, 4 * q, 6 + 4 * self.width, 2 + 4 * q])
            U_H_indices = np.array([1 + 4 * q, 5 + 4 * self.width, 3 + 4 * q, 7 + 4 * self.width])
            result_indices = np.concatenate(
                (
                    rho.indices[rho.indices < 4 * q],
                    np.array([4 + 4 * self.width, 5 + 4 * self.width, 6 + 4 * self.width, 7 + 4 * self.width]),
                    rho.indices[rho.indices > 4 * q + 3],
                )
            )
            rho.tensor = jnp.einsum(rho.tensor, rho.indices, U, U_indices, U_H, U_H_indices, result_indices)

        self.rho_map[cache_key] = rho
        return rho

    def compute_expval(self, in_data: jnp.ndarray, qubit: int):
        if qubit >= self.n_qubits:
            raise ValueError(f"qubit {qubit} is out of range with n_qubits = {self.n_qubits}")

        qubit_group = qubit // 2
        qubits = np.array([2 * qubit_group, 2 * qubit_group + 1])
        trace_qubit = 2 * int(qubit / 2) + 1 if 2 * int(qubit / 2) == qubit else 2 * int(qubit / 2)
        rho = self.compute_rho(in_data, self.depth, qubits, True)
        rho = rho.trace_rho([trace_qubit])
        return jnp.einsum(PaulyZ(), [0, 1], rho.tensor, [0, 1], [])

    def compute_all_expval(self, in_data: jnp.ndarray):
        return jnp.array([self.compute_expval(in_data, q) for q in range(self.n_qubits)]).real
