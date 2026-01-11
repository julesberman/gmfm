

import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import numpy as np


def get_turb_samples(n_samples: int, seed: int = 0):
    """
    Generate forced 2D incompressible turbulence trajectories.

    Returns:
      samples_np: (N, T, W, H, 2)  [u, v]
      times_np:   (T,)
      meta: dict with grid/domain info useful for QoIs
    """
    # Reasonable fixed defaults
    W = H = 64
    T = 101
    inner_steps = 8

    density = 1.0
    viscosity = 1e-3
    max_velocity = 2.0
    cfl_safety = 0.5

    forcing_strength = 0.15
    forcing_max_velocity = 1.0

    # 2π-periodic domain
    Lx = 2.0 * jnp.pi
    Ly = 2.0 * jnp.pi
    grid = cfd.grids.Grid((W, H), domain=((0.0, Lx), (0.0, Ly)))

    dt = cfd.equations.stable_time_step(
        max_velocity, cfl_safety, viscosity, grid)
    dt_frame = dt * inner_steps
    # times = dt_frame * jnp.arange(T)

    step_fn = cfd.funcutils.repeated(
        cfd.equations.semi_implicit_navier_stokes(
            density=density, viscosity=viscosity, dt=dt, grid=grid
        ),
        steps=inner_steps
    )

    def _add_gridvar(a: grids.GridVariable, b: grids.GridVariable, scale: jnp.ndarray) -> grids.GridVariable:
        new_array = grids.GridArray(a.data + scale * b.data, a.offset, a.grid)
        return grids.GridVariable(new_array, a.bc)

    def _add_stochastic_forcing(v, key):
        f = cfd.initial_conditions.filtered_velocity_field(
            key, grid, forcing_max_velocity)
        scale = jnp.asarray(forcing_strength, jnp.float32) * jnp.sqrt(dt_frame)
        return tuple(_add_gridvar(vi, fi, scale) for vi, fi in zip(v, f))

    def simulate_one(key):
        k_init, k_force = jax.random.split(key)

        v0 = cfd.initial_conditions.filtered_velocity_field(
            k_init, grid, max_velocity)
        u0, v0c = v0[0].data, v0[1].data  # (W,H)

        force_keys = jax.random.split(k_force, T - 1)

        def body(v, k):
            v = step_fn(v)
            v = _add_stochastic_forcing(v, k)
            return v, (v[0].data, v[1].data)

        _, (u_tail, v_tail) = jax.lax.scan(body, v0, force_keys)  # (T-1,W,H)
        u = jnp.concatenate([u0[None, ...], u_tail], axis=0)       # (T,W,H)
        vv = jnp.concatenate([v0c[None, ...], v_tail], axis=0)     # (T,W,H)
        return jnp.stack([u, vv], axis=-1)                         # (T,W,H,2)

    master = jax.random.PRNGKey(seed)
    keys = jax.random.split(master, n_samples)

    samples = jax.jit(jax.vmap(simulate_one))(keys)  # (N,T,W,H,2)

    samples_np = np.array(jax.device_get(samples))

    return samples_np
