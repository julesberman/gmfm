

import dataclasses
import jax_cfd.spectral as spectral
import jax_cfd.base.grids as grids
import seaborn as sns
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
from jax_cfd.spectral.equations import NavierStokes2D, ForcedNavierStokes2D

from gmfm.utils.tools import batchvmap


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
    W = H = 128
    T = 300
    inner_steps = 1

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

    @jax.jit
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
        return om

    master = jax.random.PRNGKey(seed)
    keys = jax.random.split(master, n_samples)

    simulate_many = batchvmap(
        simulate_one, batch_size=32, pbar=True, to_numpy=True)
    vort = simulate_many(keys)  # vel: (N,T,W,H,2), vort: (N,T,W,H)

    vort = vort[..., None]

    if only_vort:
        vort_np = np.array(jax.device_get(vort))
        return vort_np


def get_particles(tau, N, T, viscosity, max_velocity, resolution, key):

    key, new_key = jax.random.split(key)
    # physical parameters
    grid = grids.Grid((resolution, resolution), domain=(
        (0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    dt = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid)

    # setup step function using crank-nicolson runge-kutta order 4
    smooth = True  # use anti-aliasing

    # **use predefined settings for Kolmogorov flow**
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt)

    # run the simulation up until time 2
    final_time = T
    outer_steps = (final_time // dt)
    inner_steps = 1

    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

    # create an initial velocity field and compute the fft of the vorticity.
    # the spectral code assumes an fft'd vorticity for an initial state
    v0 = cfd.initial_conditions.filtered_velocity_field(
        key, grid, max_velocity, 4)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)

    _, trajectory = trajectory_fn(vorticity_hat0)

    from jax_cfd.spectral import utils as spectral_utils

    velocity_solve = spectral_utils.vorticity_to_velocity(grid)

    @jax.jit
    def reconstruct_velocities(vorticity_hat):
        vxhat, vyhat = velocity_solve(vorticity_hat)
        return (jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat))

    from jax_cfd.base.grids import GridArray

    x_offset = v0[0].array.offset
    y_offset = v0[1].array.offset

    def to_grid_array(arr, offset, grid):
        return GridArray(arr, offset, grid)

    from jax_cfd.base.interpolation import point_interpolation

    def push_particles(tau, N, trajectory, grid, dt):

        @jax.jit
        def u(x, ux, uy):
            _ux = point_interpolation(x, ux, mode='wrap')
            _uy = point_interpolation(x, uy, mode='wrap')
            return jnp.stack((_ux, _uy), axis=-1)

        ux, uy = reconstruct_velocities(trajectory[0])
        ux = to_grid_array(ux, x_offset, grid)
        uy = to_grid_array(uy, y_offset, grid)

        # xN =  jax.random.normal(new_key, shape=(N//2, 2)) +jnp.pi
        # xU = jax.random.uniform(new_key, (N - xN.shape[0], 2), minval=0, maxval=2*jnp.pi)
        # X = [ jnp.concatenate([xN, xU], axis=0)  ]

        X = [jax.random.normal(new_key, shape=(N, 2))*(1.2) + jnp.pi]
        # initial particle velocity field is equal to flow field
        V = [jax.vmap(lambda x: u(x, ux, uy))(X[-1])]

        X[0] = jnp.mod(X[0] + 2*jnp.pi, 2*jnp.pi)
        _V = V[-1]
        _X = X[-1]
        Us = []
        _i = 0
        for i, t in enumerate(trajectory):
            ux, uy = reconstruct_velocities(t)
            U_re = jnp.stack([ux, uy], -1)
            ux = to_grid_array(ux, x_offset, grid)
            uy = to_grid_array(uy, y_offset, grid)

            U = jax.vmap(lambda x: u(x, ux, uy))(_X)
            _V = _V + dt/tau * (U - _V)
            _X += dt * U
            _X = jnp.mod(_X + 2*jnp.pi, 2*jnp.pi)

            X.append(_X)
            Us.append(U_re)

        return jnp.asarray(X), jnp.asarray(Us)

    return push_particles(tau, N, trajectory, grid, dt)


@jax.jit(static_argnums=1)
def enstrophy_spectral(u: jnp.ndarray, return_vort) -> jnp.ndarray:
    """
    Spectral enstrophy on a 2π-periodic domain in both x and y.
    u: (..., H, W, 2) with (u_x, u_y). Returns (...,).
    """
    H, W = u.shape[-3], u.shape[-2]
    L = 2.0 * jnp.pi
    dx, dy = L / W, L / H

    # angular wavenumbers (for L=2π these are integers)
    kx = (2.0 * jnp.pi) * jnp.fft.fftfreq(W, d=dx)          # (W,)
    ky = (2.0 * jnp.pi) * jnp.fft.fftfreq(H, d=dy)          # (H,)
    kx = kx[None, :]                                        # (1, W)
    ky = ky[:, None]                                        # (H, 1)

    ux, uy = u[..., 0], u[..., 1]                           # (..., H, W)
    Ux = jnp.fft.fft2(ux, axes=(-2, -1))
    Uy = jnp.fft.fft2(uy, axes=(-2, -1))

    duy_dx = jnp.fft.ifft2(1j * kx * Uy, axes=(-2, -1))
    dux_dy = jnp.fft.ifft2(1j * ky * Ux, axes=(-2, -1))

    omega = jnp.real(duy_dx - dux_dy)                       # (..., H, W)
    ent = 0.5 * jnp.sum(omega * omega, axis=(-2, -1)) * dx * dy
    if return_vort:
        return omega, ent
    else:
        return ent
