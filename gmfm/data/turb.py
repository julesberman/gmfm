

import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np


def get_turb_samples(n_samples: int, seed: int = 0, only_vort: bool = True):
    """
    Generate forced 2D incompressible turbulence trajectories.

    Returns:
      vel_np:   (N, T, W, H, 2)  [u, v] (staggered components on a MAC grid)
      vort_np:  (N, T, W, H)     vorticity ω computed via JAX-CFD finite differences
      times_np: (T,)
      meta: dict with grid/domain info
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

    # Explicit periodic BCs (used to ensure consistent derivative behavior)
    bc = cfd.boundaries.HomogeneousBoundaryConditions(
        (
            (cfd.boundaries.BCType.PERIODIC, cfd.boundaries.BCType.PERIODIC),
            (cfd.boundaries.BCType.PERIODIC, cfd.boundaries.BCType.PERIODIC),
        )
    )

    dt = cfd.equations.stable_time_step(
        max_velocity, cfl_safety, viscosity, grid)
    dt_frame = dt * inner_steps

    step_fn = cfd.funcutils.repeated(
        cfd.equations.semi_implicit_navier_stokes(
            density=density, viscosity=viscosity, dt=dt, grid=grid
        ),
        steps=inner_steps
    )

    def with_bc(v):
        # overwrite bc while preserving the underlying staggered arrays/offsets
        return tuple(cfd.grids.GridVariable(vi.array, bc) for vi in v)

    def add_gridvar(a: cfd.grids.GridVariable, b: cfd.grids.GridVariable, scale: jnp.ndarray):
        # a, b are GridVariables on the same offset/grid
        new_arr = cfd.grids.GridArray(
            a.array.data + scale * b.array.data,
            grid=a.array.grid,
            offset=a.array.offset,
        )
        return cfd.grids.GridVariable(new_arr, a.bc)

    def add_stochastic_forcing(v, key):
        f = with_bc(cfd.initial_conditions.filtered_velocity_field(
            key, grid, forcing_max_velocity))
        scale = jnp.asarray(forcing_strength, jnp.float32) * jnp.sqrt(dt_frame)
        return tuple(add_gridvar(vi, fi, scale) for vi, fi in zip(v, f))

    def vorticity_from_velocity(v):
        """
        ω = ∂x v - ∂y u computed with JAX-CFD central differences on GridVariables.
        On a MAC grid this yields a consistent discrete vorticity field.
        """
        u_gv, v_gv = v
        _, du_dy = cfd.finite_differences.central_difference(u_gv)
        dv_dx, _ = cfd.finite_differences.central_difference(v_gv)
        return dv_dx.data - du_dy.data  # GridArray.data -> jnp.ndarray (W,H)

    def simulate_one(key):
        k_init, k_force = jax.random.split(key)

        v0 = with_bc(cfd.initial_conditions.filtered_velocity_field(
            k_init, grid, max_velocity))
        omega0 = vorticity_from_velocity(v0)

        force_keys = jax.random.split(k_force, T - 1)

        def body(v, k):
            v = step_fn(v)
            v = add_stochastic_forcing(v, k)
            omega = vorticity_from_velocity(v)

            u = v[0].array.data
            vv = v[1].array.data
            return v, (u, vv, omega)

        _, (u_tail, v_tail, om_tail) = jax.lax.scan(
            body, v0, force_keys)  # each (T-1,W,H)

        u = jnp.concatenate([v0[0].array.data[None, ...],
                            u_tail], axis=0)   # (T,W,H)
        vv = jnp.concatenate(
            [v0[1].array.data[None, ...], v_tail], axis=0)  # (T,W,H)
        om = jnp.concatenate([omega0[None, ...], om_tail],
                             axis=0)           # (T,W,H)

        vel = jnp.stack([u, vv], axis=-1)  # (T,W,H,2)
        return vel, om

    master = jax.random.PRNGKey(seed)
    keys = jax.random.split(master, n_samples)

    simulate_many = jax.jit(jax.vmap(simulate_one))
    vel, vort = simulate_many(keys)  # vel: (N,T,W,H,2), vort: (N,T,W,H)

    vort = vort[..., None]

    if only_vort:
        del vel
        vort_np = np.array(jax.device_get(vort))
        return vort_np
    else:
        vel_np = np.array(jax.device_get(vel))
        vort_np = np.array(jax.device_get(vort))
        return vel_np, vort_np
