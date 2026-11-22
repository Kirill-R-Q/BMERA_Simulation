"""Compatibility imports for existing BMERA scripts."""

from bmera.gates import (
    createRXX,
    createRZZ,
    createRX,
    createRZ,
    createCRX,
    createCRZ,
    createCNOT,
    InitState,
    PaulyZ,
    createBMERABlock,
    H,
)
from bmera.topology import (
    build_bmera_meta,
    _get_meta,
)
from bmera.legacy import (
    density_matrix,
    _LegacyBMERANetwork,
)
from bmera.simulator import (
    forward_single,
    forward_batch,
    forward_single_with_cache,
    forward_batch_with_cache,
    cache_profile,
    BMERANetwork,
    compute_expval,
    compute_all_expval,
    model_output,
    compute_expval_real,
    weights,
    compute_grad,
    compute_expval_and_grad,
    compute_model_grad,
)
from bmera.gradients import (
    loss_and_grad_cached,
    loss,
    loss_and_grad_autodiff_microbatched,
)
from bmera.training import (
    train_step_cached,
    train_step,
    train_loop,
    layerwise_param_mask,
)
from bmera.benchmarking import (
    benchmark_training_scaling,
)
from bmera.backends.tensorcircuit import (
    tensorcircuit_forward_single,
    tensorcircuit_forward_batch,
    tensorcircuit_loss,
    loss_and_grad_tensorcircuit,
    validate_tensorcircuit_backend,
    benchmark_tensorcircuit_backend_scaling,
)

__all__ = [
    "createRXX",
    "createRZZ",
    "createRX",
    "createRZ",
    "createCRX",
    "createCRZ",
    "createCNOT",
    "InitState",
    "PaulyZ",
    "createBMERABlock",
    "H",
    "density_matrix",
    "BMERANetwork",
    "build_bmera_meta",
    "forward_single",
    "forward_batch",
    "forward_single_with_cache",
    "forward_batch_with_cache",
    "cache_profile",
    "tensorcircuit_forward_single",
    "tensorcircuit_forward_batch",
    "tensorcircuit_loss",
    "loss_and_grad_tensorcircuit",
    "validate_tensorcircuit_backend",
    "benchmark_tensorcircuit_backend_scaling",
    "loss",
    "loss_and_grad_autodiff_microbatched",
    "loss_and_grad_cached",
    "train_step",
    "train_step_cached",
    "benchmark_training_scaling",
    "train_loop",
    "layerwise_param_mask",
    "compute_expval",
    "compute_all_expval",
    "model_output",
    "compute_expval_real",
    "compute_grad",
    "compute_expval_and_grad",
    "compute_model_grad",
    "weights",
]
