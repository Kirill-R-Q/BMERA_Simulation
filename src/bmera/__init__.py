"""JAX simulation and training for BMERA circuits."""

from .topology import (
    build_bmera_meta,
)
from .simulator import (
    forward_single,
    forward_batch,
    forward_single_with_cache,
    forward_batch_with_cache,
    cache_profile,
    BMERANetwork,
)
from .gradients import (
    loss_and_grad_cached,
    loss,
    loss_and_grad_autodiff_microbatched,
)
from .training import (
    train_step_cached,
    train_step,
    train_loop,
    layerwise_param_mask,
)

__all__ = [
    "BMERANetwork",
    "build_bmera_meta",
    "forward_single",
    "forward_batch",
    "forward_single_with_cache",
    "forward_batch_with_cache",
    "cache_profile",
    "loss",
    "loss_and_grad_autodiff_microbatched",
    "loss_and_grad_cached",
    "train_step",
    "train_step_cached",
    "train_loop",
    "layerwise_param_mask",
]
