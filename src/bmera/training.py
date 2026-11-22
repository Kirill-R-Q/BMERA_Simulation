"""Training steps and loops for BMERA models."""

from __future__ import annotations

from typing import Any, Dict
import jax.numpy as jnp
import numpy as np
from .gradients import _loss_and_grad_autodiff, loss_and_grad_cached


def train_step_cached(
    params: jnp.ndarray,
    opt_state: Dict[str, Any],
    X: jnp.ndarray,
    y: jnp.ndarray,
    meta: Dict[str, Any],
    head_weights: jnp.ndarray,
):
    lr = jnp.asarray(opt_state.get("lr", 1e-3), dtype=jnp.asarray(params).dtype)
    lval, grad_params, grad_head = loss_and_grad_cached(params, X, y, meta, head_weights)
    new_params = params - lr * grad_params
    new_head_weights = head_weights - lr * grad_head
    new_state = {**opt_state, "step": int(opt_state.get("step", 0)) + 1}
    return new_params, new_head_weights, new_state, lval


def train_step(
    params: jnp.ndarray,
    opt_state: Dict[str, Any],
    X: jnp.ndarray,
    y: jnp.ndarray,
    meta: Dict[str, Any],
    head_weights: jnp.ndarray,
    microbatch_size: int | None = None,
):
    lr = jnp.asarray(opt_state.get("lr", 1e-3), dtype=jnp.asarray(params).dtype)
    lval, grad_params, grad_head = _loss_and_grad_autodiff(
        params,
        X,
        y,
        meta,
        head_weights,
        microbatch_size=microbatch_size,
    )
    new_params = params - lr * grad_params
    new_head = head_weights - lr * grad_head
    new_state = {**opt_state, "step": int(opt_state.get("step", 0)) + 1}
    return new_params, new_head, new_state, lval


def _train_step_autodiff_full(
    params: jnp.ndarray,
    opt_state: Dict[str, Any],
    X: jnp.ndarray,
    y: jnp.ndarray,
    meta: Dict[str, Any],
    head_weights: jnp.ndarray,
    microbatch_size: int | None = None,
):
    return train_step(
        params,
        opt_state,
        X,
        y,
        meta,
        head_weights,
        microbatch_size=microbatch_size,
    )


def train_loop(
    params: jnp.ndarray,
    head_weights: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    meta: Dict[str, Any],
    steps: int,
    lr: float = 1e-3,
    grad_engine: str = "autodiff",
    param_mask: jnp.ndarray | None = None,
    microbatch_size: int | None = None,
):
    if grad_engine not in ("autodiff", "cached"):
        raise ValueError("grad_engine must be 'autodiff' or 'cached'")

    losses = []
    params_cur = jnp.asarray(params)
    head_cur = jnp.asarray(head_weights)
    state = {"lr": lr, "step": 0}

    for _ in range(steps):
        if grad_engine == "autodiff":
            lval, grad_p, grad_w = _loss_and_grad_autodiff(
                params_cur,
                X,
                y,
                meta,
                head_cur,
                microbatch_size=microbatch_size,
            )
            if param_mask is not None:
                grad_p = grad_p * param_mask
            params_cur = params_cur - lr * grad_p
            head_cur = head_cur - lr * grad_w
            state = {**state, "step": int(state["step"]) + 1}
        else:
            lval, grad_p, grad_w = loss_and_grad_cached(params_cur, X, y, meta, head_cur)
            if param_mask is not None:
                grad_p = grad_p * param_mask
            params_cur = params_cur - lr * grad_p
            head_cur = head_cur - lr * grad_w
            state = {**state, "step": int(state["step"]) + 1}
        losses.append(float(lval))

    return {
        "params": params_cur,
        "head_weights": head_cur,
        "state": state,
        "losses": np.array(losses),
    }


def layerwise_param_mask(meta: Dict[str, Any], max_unfrozen_layer: int) -> jnp.ndarray:
    depth = int(meta["depth"])
    width = int(meta["n_qubits"])
    n_params = int(meta["n_per_block"])
    max_layer = int(np.clip(max_unfrozen_layer, 1, depth))

    mask = np.zeros((depth * width, n_params), dtype=np.float32)
    mask[: max_layer * width, :] = 1.0
    return jnp.asarray(mask)
