"""Comparison and gradient diagnostics for spin-model datasets."""

from pathlib import Path

from . import snapshots as qsb

SNAPSHOT_BENCHMARK_MODELS = qsb.SNAPSHOT_MODELS
SNAPSHOT_SCREEN_QUBITS = (8,)
SNAPSHOT_CONFIRM_QUBITS = (8, 16)
SNAPSHOT_CLASSICAL_MODELS = (
    'logistic',
    'rbf_svm',
    'random_forest',
    'matched_mlp',
)
SNAPSHOT_FINAL_SEEDS = (0, 1, 2, 3, 4)
SNAPSHOT_CACHE_DIR = Path('quantum_snapshot_runs/cache')
SNAPSHOT_RESULT_DIR = Path('quantum_snapshot_runs/results')
SNAPSHOT_SCREEN_STATES = dict(qsb.SCREEN_STATE_COUNTS)
SNAPSHOT_CONFIRM_STATES = dict(qsb.CONFIRM_STATE_COUNTS)
SNAPSHOT_SCREEN_SHOTS = 128
SNAPSHOT_CONFIRM_SHOTS = 256


def build_snapshot_dataset_suite(profile='screen', models=SNAPSHOT_BENCHMARK_MODELS, *,
                                 cache_dir=SNAPSHOT_CACHE_DIR, seed=20260611):
    if profile == 'screen':
        qubits = SNAPSHOT_SCREEN_QUBITS
        state_counts = SNAPSHOT_SCREEN_STATES
        snapshots_per_state = SNAPSHOT_SCREEN_SHOTS
    elif profile == 'confirmation':
        qubits = SNAPSHOT_CONFIRM_QUBITS
        state_counts = SNAPSHOT_CONFIRM_STATES
        snapshots_per_state = SNAPSHOT_CONFIRM_SHOTS
    elif profile == 'smoke':
        qubits = (8,)
        state_counts = {name: 1 for name in qsb.SNAPSHOT_SPLITS}
        snapshots_per_state = 4
    else:
        raise ValueError("profile must be 'smoke', 'screen', or 'confirmation'")

    datasets = {}
    for model_name in models:
        for n_qubits in qubits:
            print(
                f'Generating/loading {model_name}, n={n_qubits}, '
                f'profile={profile}'
            )
            dataset = qsb.generate_snapshot_dataset(
                model_name,
                n_qubits,
                state_counts=state_counts,
                snapshots_per_state=snapshots_per_state,
                cache_dir=cache_dir,
                seed=seed,
            )
            qsb.validate_snapshot_dataset(dataset)
            datasets[(model_name, n_qubits)] = dataset
    return datasets


def run_single_snapshot_comparison(dataset, smoke=False):
    n_qubits = int(dataset['metadata']['n_qubits'])
    if smoke:
        bmera_result = qsb.tune_bmera_snapshot_classifier(
            dataset,
            learning_rates=(0.01,),
            init_scales=(0.1,),
            final_seeds=(0,),
            max_steps=1,
            batch_size=4,
            eval_every=1,
            patience=1,
        )
        final_seeds = (0,)
    else:
        bmera_result = qsb.tune_bmera_snapshot_classifier(
            dataset,
            final_seeds=SNAPSHOT_FINAL_SEEDS,
            max_steps=300,
            batch_size=128,
            eval_every=10,
            patience=8,
        )
        final_seeds = SNAPSHOT_FINAL_SEEDS

    parameter_count = qsb.bmera_parameter_count(n_qubits)
    model_results = {'bmera': bmera_result}
    for model_name in SNAPSHOT_CLASSICAL_MODELS:
        print(
            f"Classical baseline | dataset={dataset['metadata']['model']} | "
            f'n={n_qubits} | model={model_name}'
        )
        model_results[model_name] = qsb.tune_classical_baseline(
            dataset,
            model_name,
            bmera_parameter_count=parameter_count,
            final_seeds=final_seeds,
        )
    return {
        'dataset': dataset['metadata']['model'],
        'n_qubits': n_qubits,
        'models': model_results,
    }


def run_snapshot_benchmark_suite(datasets, smoke=False):
    results = []
    for key in sorted(datasets):
        print(f'Benchmarking snapshot dataset {key}')
        results.append(
            run_single_snapshot_comparison(datasets[key], smoke=smoke)
        )
    ranking = qsb.rank_dataset_advantages(
        results,
        n_bootstrap=100 if smoke else 10000,
    )
    return results, ranking


def print_snapshot_comparison(results, ranking):
    print(
        'dataset | qubits | model | critical state balanced accuracy '
        '(mean +/- std)'
    )
    for dataset_result in results:
        for model_name, result in dataset_result['models'].items():
            summary = qsb.summarize_model_result(result)
            mean = summary['critical_test_state_balanced_accuracy_mean']
            std = summary['critical_test_state_balanced_accuracy_std']
            print(
                f"{dataset_result['dataset']:>13s} | "
                f"{dataset_result['n_qubits']:>6d} | "
                f"{model_name:>13s} | {mean:.3f} +/- {std:.3f}"
            )
    print('\nBMERA difference versus the classical model selected on validation')
    for row in ranking:
        support = 'supported' if row['supported_advantage'] else 'not supported'
        print(
            f"{row['dataset']:>13s} n={row['n_qubits']:2d} | "
            f"best classical={row['best_classical']:>13s} | "
            f"gap={row['difference']:+.3f} "
            f"[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}] | {support}"
        )


# Multi-model measurement-snapshot path diagnostics.
SNAPSHOT_PATH_MODELS = qsb.SNAPSHOT_MODELS
SNAPSHOT_PATH_QUBITS = (8, 16)
SNAPSHOT_PATH_STEPS = (0, 1, 2, 3, 5, 8, 13, 21, 34, 50, 75, 100)
SNAPSHOT_PATH_EVAL_BATCH_SIZE = 128
SNAPSHOT_PATH_EVAL_BATCHES = 8
SNAPSHOT_PATH_TRAIN_BATCH_SIZE = 128


def run_all_snapshot_path_diagnostics(smoke=False):
    if smoke:
        models = ('tfim',)
        qubits = (8,)
        state_counts = {name: 1 for name in qsb.SNAPSHOT_SPLITS}
        snapshots_per_state = 4
        steps = (0, 1)
        eval_batch_size = 4
        eval_batches = 2
        train_batch_size = 4
    else:
        models = SNAPSHOT_PATH_MODELS
        qubits = SNAPSHOT_PATH_QUBITS
        state_counts = SNAPSHOT_SCREEN_STATES
        snapshots_per_state = SNAPSHOT_SCREEN_SHOTS
        steps = SNAPSHOT_PATH_STEPS
        eval_batch_size = SNAPSHOT_PATH_EVAL_BATCH_SIZE
        eval_batches = SNAPSHOT_PATH_EVAL_BATCHES
        train_batch_size = SNAPSHOT_PATH_TRAIN_BATCH_SIZE

    parameter_rows = []
    group_rows = []
    global_rows = []
    heatmaps = {}
    devices = {}
    for model_name in models:
        for n_qubits in qubits:
            dataset = qsb.generate_snapshot_dataset(
                model_name,
                n_qubits,
                state_counts=state_counts,
                snapshots_per_state=snapshots_per_state,
                cache_dir=SNAPSHOT_CACHE_DIR,
            )
            print(f'Gradient path | model={model_name} | n={n_qubits}')
            result = qsb.run_snapshot_path_diagnostics(
                dataset,
                snapshot_steps=steps,
                eval_batch_size=eval_batch_size,
                eval_batches=eval_batches,
                train_batch_size=train_batch_size,
                learning_rate=0.01,
                init_scale=0.1,
                seed=0,
            )
            parameter_rows.extend(result['parameter_rows'])
            group_rows.extend(result['group_rows'])
            global_rows.extend(result['global_rows'])
            heatmaps.update(result['heatmaps'])
            devices[(model_name, n_qubits)] = str(result['device'])
    return parameter_rows, group_rows, global_rows, heatmaps, devices


def print_snapshot_path_table(rows):
    if not rows:
        print('No snapshot path diagnostics.')
        return
    print(
        'dataset | qubits | step | loss | normalized RMS | mean variance | '
        'median parameter SNR | vector SNR | mean abs'
    )
    for row in sorted(
        rows,
        key=lambda item: (item['dataset'], item['n_qubits'], item['step']),
    ):
        print(
            f"{row['dataset']:>13s} | {row['n_qubits']:>6d} | "
            f"{row['step']:>4d} | {row['eval_loss_mean']:.4f} | "
            f"{row['normalized_gradient_rms']:.3e} | "
            f"{row['mean_parameter_variance']:.3e} | "
            f"{row['median_parameter_snr']:.3e} | "
            f"{row['vector_snr']:.3e} | "
            f"{row['mean_absolute_gradient']:.3e}"
        )
