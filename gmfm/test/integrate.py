
"""Sampling utilities used during integration tests or quick eval runs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import lax

from gmfm.config.config import Config


def sample_trajectories(
    x0: jnp.ndarray,
    params,
    apply_fn,
    n_steps,
    key,
    *,
    sigma: float = 0.0,
    boundary: str = "none",   # keep in [-1, 1] during integration
) -> jnp.ndarray:
    """
    Euler integrate from t=0 to t=1 (inclusive) using lax.scan.
    Returns an array of shape (n_steps, *x0.shape) with traj[0] == x0.

    boundary:
      - "clip":      x <- clip(x, -1, 1)
      - "periodize": x <- ((x + 1) % 2) - 1   (wrap to [-1, 1))
      - "none":      no projection
    """

    batch_size = x0.shape[0]
    ts = jnp.linspace(0.0, 1.0, n_steps, dtype=x0.dtype)
    dt = ts[1] - ts[0]

    if boundary is not None:
        if boundary == "clip":
            def proj(x): return jnp.clip(x, -1.0, 1.0)
        elif boundary == "periodize":
            def proj(x): return ((x + 1.0) % 2.0) - 1.0
    else:
        def proj(x): return x

    def step_sde(carry, ti):
        x, k = carry
        k, subk = jax.random.split(k)
        t_batch = jnp.full((batch_size, 1), ti, dtype=x.dtype)
        v = apply_fn(params, x, t_batch)
        eps = jax.random.normal(subk, shape=x.shape, dtype=x.dtype)
        x_next = proj(x + v * dt + (sigma * jnp.sqrt(dt)) * eps)
        return (x_next, k), x_next

    def step_ode(carry, ti):
        x, k = carry
        t_batch = jnp.full((batch_size, 1), ti, dtype=x.dtype)
        v = apply_fn(params, x, t_batch)
        x_next = proj(x + v * dt)
        return (x_next, k), x_next

    x0 = proj(x0)

    if sigma > 0.0:
        step = step_sde
    else:
        step = step_ode

    (_, _), xs = lax.scan(step, (x0, key), ts[:-1])
    return jnp.concatenate([x0[None, ...], xs], axis=0)


def sample_model(cfg: Config, apply_fn, opt_params, x_0, sigma, n_steps, key):

    boundary = cfg.integrate.boundary

    traj = sample_trajectories(
        x_0, opt_params, apply_fn, n_steps, key, sigma=sigma, boundary=boundary)
    traj = np.asarray(traj)

    traj = rearrange(traj, 'T N ... -> N T ...')
    return traj
