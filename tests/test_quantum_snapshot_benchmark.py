import tempfile
import unittest
from pathlib import Path

import numpy as np

from bmera.experiments import snapshots as qsb


class QuantumSnapshotBenchmarkTests(unittest.TestCase):
    def test_bmera_parameter_budget_uses_active_blocks(self):
        self.assertEqual(qsb.bmera_parameter_count(8), 88)
        self.assertEqual(qsb.bmera_parameter_count(8, allocated=True), 104)

    def test_hamiltonians_are_hermitian_and_ground_states_converge(self):
        for model in qsb.SNAPSHOT_MODELS:
            hamiltonian = qsb.build_hamiltonian(model, 4, 0.7)
            difference = hamiltonian - hamiltonian.getH()
            self.assertLess(np.linalg.norm(difference.toarray()), 1e-12)
            energy, state, residual = qsb.ground_state(model, 4, 0.7)
            self.assertTrue(np.isfinite(energy))
            self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=10)
            self.assertLess(residual, 1e-8)

    def test_known_physical_limits(self):
        _, tfim_low, _ = qsb.ground_state("tfim", 6, 0.2)
        _, tfim_high, _ = qsb.ground_state("tfim", 6, 2.0)
        low_obs = qsb.physical_observables("tfim", tfim_low, 6)
        high_obs = qsb.physical_observables("tfim", tfim_high, 6)
        self.assertGreater(low_obs["mean_nn_zz"], high_obs["mean_nn_zz"])
        self.assertGreater(high_obs["mean_x"], low_obs["mean_x"])

        _, cluster_low, _ = qsb.ground_state("cluster_ising", 6, 0.2)
        _, cluster_high, _ = qsb.ground_state("cluster_ising", 6, 2.0)
        low_obs = qsb.physical_observables("cluster_ising", cluster_low, 6)
        high_obs = qsb.physical_observables("cluster_ising", cluster_high, 6)
        self.assertGreater(low_obs["mean_cluster"], high_obs["mean_cluster"])
        self.assertGreater(high_obs["mean_x"], low_obs["mean_x"])

        _, xxz_low, _ = qsb.ground_state("xxz", 6, 0.5)
        _, xxz_high, _ = qsb.ground_state("xxz", 6, 1.8)
        low_obs = qsb.physical_observables("xxz", xxz_low, 6)
        high_obs = qsb.physical_observables("xxz", xxz_high, 6)
        self.assertGreater(
            high_obs["staggered_structure_factor"],
            low_obs["staggered_structure_factor"],
        )

    def test_basis_sampling_encoding_and_reproducibility(self):
        masks = qsb.balanced_basis_masks(8, 16, seed=3)
        self.assertTrue(np.all(masks.sum(axis=0) == 8))
        state = np.zeros(1 << 8, dtype=np.complex128)
        state[0] = 1.0
        outcomes1 = qsb.sample_measurement_snapshots(state, masks, seed=4)
        outcomes2 = qsb.sample_measurement_snapshots(state, masks, seed=4)
        np.testing.assert_array_equal(outcomes1, outcomes2)
        self.assertTrue(np.all(outcomes1[masks == 0] == 1))
        angles = qsb.encode_snapshot_angles(masks, outcomes1)
        allowed = np.asarray([-np.pi / 4, 0.0, np.pi / 4, np.pi / 2])
        self.assertTrue(np.all(np.min(np.abs(angles[..., None] - allowed), axis=-1) < 1e-6))

    def test_dataset_cache_and_state_disjointness(self):
        counts = {name: 1 for name in qsb.SNAPSHOT_SPLITS}
        with tempfile.TemporaryDirectory() as temporary:
            first = qsb.generate_snapshot_dataset(
                "tfim",
                4,
                state_counts=counts,
                snapshots_per_state=4,
                seed=7,
                cache_dir=temporary,
            )
            second = qsb.generate_snapshot_dataset(
                "tfim",
                4,
                state_counts=counts,
                snapshots_per_state=4,
                seed=7,
                cache_dir=temporary,
            )
            qsb.validate_snapshot_dataset(first)
            for split_name in qsb.SNAPSHOT_SPLITS:
                np.testing.assert_array_equal(first[split_name]["X"], second[split_name]["X"])
                np.testing.assert_array_equal(
                    first[split_name]["state_id"],
                    second[split_name]["state_id"],
                )
            self.assertTrue(Path(first["metadata"]["dataset_path"]).exists())

    def test_grouped_metrics(self):
        split = {
            "state_id": np.asarray(["a", "a", "b", "b"]),
            "y": np.asarray([-1.0, -1.0, 1.0, 1.0]),
            "parameter": np.asarray([0.5, 0.5, 1.5, 1.5]),
        }
        metrics = qsb.evaluate_snapshot_scores(
            np.asarray([-0.8, -0.2, 0.3, 0.9]),
            split,
        )
        self.assertEqual(metrics["snapshot_balanced_accuracy"], 1.0)
        self.assertEqual(metrics["state_balanced_accuracy"], 1.0)
        self.assertEqual(metrics["state_auroc"], 1.0)

    def test_all_classical_baselines_smoke(self):
        counts = {name: 1 for name in qsb.SNAPSHOT_SPLITS}
        with tempfile.TemporaryDirectory() as temporary:
            dataset = qsb.generate_snapshot_dataset(
                "cluster_ising",
                8,
                state_counts=counts,
                snapshots_per_state=4,
                seed=9,
                cache_dir=temporary,
            )
            parameter_count = qsb.bmera_parameter_count(8)
            for model_name in ("logistic", "rbf_svm", "random_forest", "matched_mlp"):
                result = qsb.tune_classical_baseline(
                    dataset,
                    model_name,
                    bmera_parameter_count=parameter_count,
                    final_seeds=(0,),
                )
                self.assertEqual(result["model"], model_name)
                self.assertEqual(len(result["runs"]), 1)
                self.assertTrue(
                    np.isfinite(
                        result["runs"][0]["splits"]["critical_test"][
                            "state_balanced_accuracy"
                        ]
                    )
                )

    def test_bmera_gradient_smoke(self):
        counts = {name: 1 for name in qsb.SNAPSHOT_SPLITS}
        with tempfile.TemporaryDirectory() as temporary:
            dataset = qsb.generate_snapshot_dataset(
                "tfim",
                8,
                state_counts=counts,
                snapshots_per_state=4,
                seed=11,
                cache_dir=temporary,
            )
            result = qsb.run_snapshot_path_diagnostics(
                dataset,
                snapshot_steps=(0, 1),
                eval_batch_size=4,
                eval_batches=2,
                train_batch_size=4,
                prefer_gpu=False,
            )
            self.assertEqual([row["step"] for row in result["global_rows"]], [0, 1])
            self.assertTrue(result["parameter_rows"])
            self.assertTrue(result["group_rows"])
            self.assertFalse(any(row["layer"] == 0 and row["color"] == "red" for row in result["group_rows"]))
            for row in result["global_rows"]:
                self.assertEqual(row["n_parameters"], 80)
                self.assertTrue(np.isfinite(row["vector_snr"]))
                self.assertTrue(np.isfinite(row["normalized_gradient_rms"]))


if __name__ == "__main__":
    unittest.main()
