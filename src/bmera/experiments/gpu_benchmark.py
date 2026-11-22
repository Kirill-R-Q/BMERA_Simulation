"""GPU gradient throughput sweeps."""

import jax
import jax.numpy as jnp
import numpy as np
from bmera import build_bmera_meta, loss_and_grad_autodiff_microbatched
from bmera.benchmarking import benchmark_callable, benchmark_context

import subprocess

GPU_SWEEP_QUBITS = (256,)
GPU_BATCH_SWEEP = tuple(2**k for k in range(0, 13))  # 1 ... 4096
GPU_BATCH_REPEATS = 5
GPU_BATCH_WARMUP = 2
OPTIMAL_THROUGHPUT_FRACTION = 0.98

def get_jax_gpu_devices():
    return [device for device in jax.devices() if device.platform in ('gpu', 'cuda', 'rocm')]

def probe_gpu_runtime():
    platforms = sorted({device.platform for device in jax.devices()})
    gpu_devices = get_jax_gpu_devices()
    diagnostic = {
        'jax_platforms': platforms,
        'jax_gpu_devices': gpu_devices,
        'nvidia_smi_ok': False,
        'nvidia_smi_summary': None,
    }
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10, check=False)
        stdout = (result.stdout or '').strip()
        stderr = (result.stderr or '').strip()
        diagnostic['nvidia_smi_ok'] = result.returncode == 0
        summary = next((line.strip() for line in stdout.splitlines() if 'Driver Version:' in line), None)
        if summary is None:
            text = stdout if stdout else stderr
            summary = text.splitlines()[0] if text else 'nvidia-smi produced no output.'
        diagnostic['nvidia_smi_summary'] = summary
    except Exception as exc:
        diagnostic['nvidia_smi_summary'] = repr(exc)
    return diagnostic

def gpu_sweep_unavailable_reason(diagnostic):
    platforms = diagnostic['jax_platforms']
    if diagnostic['nvidia_smi_ok']:
        return (
            'NVIDIA GPU detected by nvidia-smi, but this notebook kernel has no JAX GPU backend. '
            f'Visible JAX platforms: {platforms}. Install a CUDA-enabled jaxlib in this Python environment and restart the kernel.'
        )
    return (
        'No JAX GPU backend is available for the requested GPU sweep. '
        f'Visible JAX platforms: {platforms}. nvidia-smi: {diagnostic["nvidia_smi_summary"]}'
    )

def benchmark_autodiff_grad_batchsize_gpu(
    n_qubits,
    batch_sizes=GPU_BATCH_SWEEP,
    repeats=GPU_BATCH_REPEATS,
    warmup=GPU_BATCH_WARMUP,
    seed=42,
    microbatch_size=None,
):
    diagnostic = probe_gpu_runtime()
    gpu_devices = diagnostic['jax_gpu_devices']
    if not gpu_devices:
        reason = gpu_sweep_unavailable_reason(diagnostic)
        print(reason)
        return [], [{'n_qubits': int(n_qubits), 'batch_size': None, 'error': reason}]

    gpu_device = gpu_devices[0]
    print(f'Benchmarking {n_qubits}-qubit gradient sweep on GPU device: {gpu_device}')
    if diagnostic['nvidia_smi_ok'] and diagnostic['nvidia_smi_summary']:
        print(diagnostic['nvidia_smi_summary'])

    rng = np.random.default_rng(seed + int(n_qubits))
    type_ = 1
    meta = build_bmera_meta(n_qubits, type_)
    param_shape = (int(meta['depth']) * n_qubits, int(meta['n_per_block']))

    rows = []
    failures = []

    for batch_size in batch_sizes:
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()

        params = jnp.asarray(rng.uniform(-0.5, 0.5, size=param_shape), dtype=jnp.float32)
        X = jnp.asarray(rng.uniform(-0.5, 0.5, size=(batch_size, n_qubits)), dtype=jnp.float32)
        y = jnp.asarray(rng.uniform(-1.0, 1.0, size=(batch_size,)), dtype=jnp.float32)
        head = jnp.asarray(rng.uniform(-0.5, 0.5, size=(n_qubits,)), dtype=jnp.float32)

        params = jax.device_put(params, gpu_device)
        X = jax.device_put(X, gpu_device)
        y = jax.device_put(y, gpu_device)
        head = jax.device_put(head, gpu_device)

        effective_microbatch = batch_size if microbatch_size is None or microbatch_size <= 0 or microbatch_size >= batch_size else int(microbatch_size)

        def grad_call():
            return loss_and_grad_autodiff_microbatched(params, X, y, meta, head, effective_microbatch)

        try:
            timing = benchmark_callable(grad_call, repeats, warmup)
            compile_plus_first_ms = timing['compile_plus_first_ms']
            grad_ms = timing['steady_mean_ms']
            grad_std_ms = timing['steady_std_ms']
            samples_per_second = 1000.0 * batch_size / grad_ms

            rows.append({
                'n_qubits': int(n_qubits),
                'batch_size': int(batch_size),
                'microbatch_size': microbatch_size,
                'effective_microbatch_size': int(effective_microbatch),
                'grad_compile_plus_first_ms': float(compile_plus_first_ms),
                'grad_steady_ms': grad_ms,
                **benchmark_context(params.device, params.dtype, repeats, warmup, seed),
                'grad_steady_std_ms': grad_std_ms,
                'grad_ms_per_sample': grad_ms / batch_size,
                'grad_samples_per_second': samples_per_second,
            })
            print(
                f'batch={batch_size:5d} | grad={grad_ms:9.3f} ms '
                f'| per_sample={grad_ms / batch_size:8.4f} ms '
                f'| throughput={samples_per_second:9.1f} samples/s '
                f'| micro={effective_microbatch:4d}'
            )
        except Exception as exc:
            failures.append({'n_qubits': int(n_qubits), 'batch_size': int(batch_size), 'error': repr(exc)})
            print(f'n={n_qubits:2d} | batch={batch_size:5d} failed: {exc!r}')
            break

    return rows, failures


def summarize_gpu_batch_sweep(rows, failures, throughput_fraction=OPTIMAL_THROUGHPUT_FRACTION):
    if not rows:
        message = failures[0]['error'] if failures else 'No GPU benchmark rows were collected.'
        print(message)
        return {
            'status': 'no_gpu_data',
            'message': message,
        }

    peak = max(rows, key=lambda r: r['grad_samples_per_second'])
    cutoff = throughput_fraction * peak['grad_samples_per_second']
    near_optimal = [r for r in rows if r['grad_samples_per_second'] >= cutoff]
    max_near_optimal = max(near_optimal, key=lambda r: r['batch_size'])
    first = rows[0]
    largest_tested = rows[-1]
    first_failed_batch = next((int(f['batch_size']) for f in failures if f.get('batch_size') is not None), None)

    n_qubits = int(rows[0]['n_qubits'])

    summary = {
        'status': 'ok',
        'n_qubits': n_qubits,
        'grad_engine': 'autodiff',
        'throughput_fraction': float(throughput_fraction),
        'peak_batch_size': int(peak['batch_size']),
        'peak_grad_steady_ms': float(peak['grad_steady_ms']),
        'peak_grad_samples_per_second': float(peak['grad_samples_per_second']),
        'max_near_optimal_batch_size': int(max_near_optimal['batch_size']),
        'max_near_optimal_grad_steady_ms': float(max_near_optimal['grad_steady_ms']),
        'max_near_optimal_grad_samples_per_second': float(max_near_optimal['grad_samples_per_second']),
        'speedup_vs_batch_1_at_peak': float(peak['grad_samples_per_second'] / first['grad_samples_per_second']),
        'largest_successful_batch_size': int(largest_tested['batch_size']),
        'first_failed_batch_size': first_failed_batch,
        'failure_messages': [f['error'] for f in failures],
    }

    print(
        f'{n_qubits}-qubit exact throughput peak: '
        f"batch={summary['peak_batch_size']}, "
        f"{summary['peak_grad_samples_per_second']:.1f} samples/s, "
        f"grad={summary['peak_grad_steady_ms']:.3f} ms"
    )
    print(
        f"{n_qubits}-qubit maximum near-optimal batch (>={throughput_fraction:.0%} of peak): "
        f"batch={summary['max_near_optimal_batch_size']}, "
        f"{summary['max_near_optimal_grad_samples_per_second']:.1f} samples/s, "
        f"grad={summary['max_near_optimal_grad_steady_ms']:.3f} ms"
    )

    if summary['peak_batch_size'] == summary['largest_successful_batch_size'] and not failures:
        print('Peak throughput occurred at the largest tested batch. Increase GPU_BATCH_SWEEP to check for a later plateau or memory limit.')
    if failures:
        print(f"First failed batch: {summary['first_failed_batch_size']}")

    return summary
