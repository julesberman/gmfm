
import random

import jax
import jax.numpy as jnp
import jax.random
import lineax
from diffrax import (
    ControlTerm,
    Euler,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)
from jax import jit, vmap


def get_ic_lorenz9d(key, noise=5e-2):
    n_particles = 9
    var = noise
    ic = jax.random.normal(key, (n_particles,)) * var
    return ic


def get_lorenz9d(mu, noise=2e-2):

    def drift(t, y, *args):
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = y

        a = 0.5
        s = 0.5
        b1 = 4 * (1 + a**2) / (1 + 2 * a**2)
        b2 = (1 + 2 * a**2) / (2 * (1 + a**2))
        b3 = 2 * ((1 - a**2) / (1 + a**2))
        b4 = a**2 / (1 + a**2)
        b5 = (8 * a**2) / (1 + 2 * a**2)
        b6 = 4 / (1 + 2 * a**2)

        r = mu

        c1_dot = -s * b1 * c1 - c2 * c4 + b4 * c4**2 + b3 * c3 * c5 - s * b2 * c7
        c2_dot = -s * c2 + c1 * c4 - c2 * c5 + c4 * c5 - s * c9 / 2
        c3_dot = -s * b1 * c3 + c2 * c4 - b4 * c2**2 - b3 * c1 * c5 + s * b2 * c8
        c4_dot = -s * c4 - c2 * c3 - c2 * c5 + c4 * c5 + s * c9 / 2
        c5_dot = -s * b5 * c5 + c2**2 / 2 - c4**2 / 2
        c6_dot = -b6 * c6 + c2 * c9 - c4 * c9
        c7_dot = -b1 * c7 - r * c1 + 2 * c5 * c8 - c4 * c9
        c8_dot = -b1 * c8 + r * c3 - 2 * c5 * c7 + c2 * c9
        c9_dot = -c9 - r * c2 + r * c4 - 2 * c2 * c6 + 2 * c4 * c6 + c4 * c7 - c2 * c8
        return jnp.asarray(
            [c1_dot, c2_dot, c3_dot, c4_dot, c5_dot, c6_dot, c7_dot, c8_dot, c9_dot]
        )

    def diffusion(t, y, *args):

        return jnp.ones_like(y) * noise

    return drift, diffusion


def solve_sde(drift, diffusion, t_eval, get_ic, n_samples, dt=1e-2, key=None):
    t_eval = jnp.asarray(t_eval)

    @jit
    def solve_single(key):
        ikey, skey = jax.random.split(key)
        y0 = get_ic(ikey)
        sol = solve_sde_ic(y0, skey, t_eval, dt, drift, diffusion)
        return sol

    if key is None:
        key = jax.random.PRNGKey(random.randint(0, 1e6))
    keys = jax.random.split(key, num=n_samples)
    solve_single = vmap(solve_single)

    sols = solve_single(keys)

    return sols


def solve_sde_ic(y0, key, t_eval, dt, drift, diffusion):
    t0, t1 = t_eval[0], t_eval[-1]
    brownian_motion = VirtualBrownianTree(
        t0, t1, tol=1e-3, shape=y0.shape, key=key)

    def diag_diffusion(*args):
        return lineax.DiagonalLinearOperator(diffusion(*args))
    diffusion_term = ControlTerm(diag_diffusion, brownian_motion)
    terms = MultiTerm(ODETerm(drift), diffusion_term)
    solver = Euler()
    saveat = SaveAt(ts=t_eval)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt,
                      y0=y0, saveat=saveat, max_steps=int(1e6))

    return sol.ys


def get_lz9_data(n_samples, t_eval, key):
    mu = 13.65  # 14.05
    drift, diffusion = get_lorenz9d(mu, noise=5e-2)
    return solve_sde(drift, diffusion, t_eval, get_ic_lorenz9d, n_samples, dt=1e-2, key=key)
