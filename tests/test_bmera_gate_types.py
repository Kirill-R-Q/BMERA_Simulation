import unittest
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pennylane as qml

from bmera import (
    BMERANetwork,
    build_bmera_meta,
    forward_batch,
    forward_single,
    forward_single_with_cache,
    loss_and_grad_autodiff_microbatched,
    loss_and_grad_cached,
)
from bmera.backends.pennylane import (
    BMERA_CNOT_block,
    BMERA_CRX_block,
    BMERA_CRZ_block,
    pennylane_bmera_expvals,
)
from bmera.backends.tensorcircuit import validate_tensorcircuit_backend
from bmera.gates import createBMERABlock


try:
    version("tensorcircuit-ng")
except PackageNotFoundError:
    HAS_TENSORCIRCUIT = False
else:
    HAS_TENSORCIRCUIT = True


class BmeraGateTypeTests(unittest.TestCase):
    def test_controlled_rotations_match_pennylane_matrices(self):
        wires = [0, 1]
        angles = np.array([0.21, -0.42, 0.37, 0.92, -0.67], dtype=np.float32)
        references = (BMERA_CNOT_block, BMERA_CRX_block, BMERA_CRZ_block)
        for block_type, reference in enumerate(references, start=1):
            with self.subTest(block_type=block_type):
                params = angles[:4] if block_type == 1 else angles
                expected = qml.matrix(reference, wire_order=wires)(params, wires)
                block = createBMERABlock(block_type, params)
                actual = block.transpose(0, 2, 1, 3).reshape(4, 4)
                np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)

    def test_single_and_batch_match_reference_in_both_parameter_layouts(self):
        rng = np.random.default_rng(18)
        X = rng.standard_normal((2, 4), dtype=np.float32)
        for block_type in (1, 2, 3):
            with self.subTest(block_type=block_type):
                meta = build_bmera_meta(4, block_type)
                flat = rng.standard_normal((8, meta["n_per_block"]), dtype=np.float32)
                layered = flat.reshape(2, 4, -1)
                expected = np.stack([pennylane_bmera_expvals(x, 4, block_type, flat) for x in X])
                np.testing.assert_allclose(forward_batch(flat, X, meta), expected, atol=2e-6)
                np.testing.assert_allclose(forward_batch(layered, X, meta), expected, atol=2e-6)
                np.testing.assert_allclose(forward_single(layered, X[0], meta), expected[0], atol=2e-6)
                cached, _ = forward_single_with_cache(layered, X[0], meta)
                np.testing.assert_allclose(cached, expected[0], atol=2e-6)

    def test_rejects_invalid_shapes_and_qubit_index(self):
        meta = build_bmera_meta(4, 1)
        params = np.zeros((8, 4), dtype=np.float32)
        x = np.zeros(4, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "params must have shape"):
            forward_single(params[:1], x, meta)
        with self.assertRaisesRegex(ValueError, "params must have shape"):
            forward_single(np.zeros((2, 4, 5)), x, meta)
        with self.assertRaisesRegex(ValueError, "x must have shape"):
            forward_single(params, x[:2], meta)
        with self.assertRaisesRegex(ValueError, "X must have shape"):
            forward_batch(params, x, meta)
        with self.assertRaisesRegex(ValueError, "same nonzero number of samples"):
            loss_and_grad_cached(params, x[None, :], np.zeros((1, 1)), meta, x)
        with self.assertRaisesRegex(ValueError, "head_weights must have shape"):
            loss_and_grad_autodiff_microbatched(params, x[None, :], np.zeros(1), meta, x[:2], 1)
        for qubit in (-1, 4):
            with self.assertRaisesRegex(ValueError, "out of range"):
                BMERANetwork(4, params=params).compute_expval(x, qubit)
        for n_qubits in (0, -1, 3, 4.5):
            with self.assertRaises(ValueError):
                build_bmera_meta(n_qubits, 1)

    @unittest.skipUnless(HAS_TENSORCIRCUIT, "TensorCircuit-NG is optional")
    def test_tensorcircuit_parity_preserves_numpy(self):
        reshape = np.reshape
        rows = validate_tensorcircuit_backend(n_qubits_list=(4,), batch_size=1)
        for row in rows:
            with self.subTest(block_type=row["type"]):
                self.assertLess(row["forward_max_abs_error"], 5e-6)
                self.assertLess(row["loss_abs_error"], 5e-6)
                self.assertLess(row["grad_max_abs_error"], 5e-6)
        self.assertIs(np.reshape, reshape)
        self.assertEqual(np.reshape(np.arange(4), (2, 2), copy=True).shape, (2, 2))

    def test_cached_and_microbatched_gradients_match_autodiff(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((3, 4), dtype=np.float32)
        y = rng.standard_normal(3, dtype=np.float32)
        head_weights = rng.standard_normal(4, dtype=np.float32)

        for block_type, n_params in ((1, 4), (2, 5), (3, 5)):
            with self.subTest(block_type=block_type):
                meta = build_bmera_meta(4, block_type)
                params = rng.standard_normal((8, n_params), dtype=np.float32)
                expected = loss_and_grad_autodiff_microbatched(params, X, y, meta, head_weights, None)
                cached = loss_and_grad_cached(params, X, y, meta, head_weights)
                microbatched = loss_and_grad_autodiff_microbatched(params, X, y, meta, head_weights, 2)

                for result in (cached, microbatched):
                    for actual, target in zip(result, expected):
                        np.testing.assert_allclose(actual, target, atol=5e-6, rtol=5e-6)
