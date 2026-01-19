import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import vmap
from jax.numpy.fft import fft2, fftfreq, ifft2

from gmfm.utils.tools import batchvmap, normalize


def solve_wave_equation(Tend, dt, N, ic_field, speed):
    """
    Solves the 2D wave equation using a spectral method.

    Parameters:
    Tend (float): End time of the simulation.
    dt (float): Time step.
    N (int): Number of grid points in x and y directions.
    ic_fn (callable): Function of (x, y) giving initial condition u(x, y, 0).
    speed_fn (callable): Function of (x, y) giving wave speed c(x, y).

    Returns:
    u_n (jax.numpy.ndarray): The solution u(x, y, Tend) at the final time.
    """
    L = 2 * jnp.pi  # Domain size

    dx = L / N
    x = jnp.linspace(0, L - dx, N)
    y = jnp.linspace(0, L - dx, N)
    m_grids = jnp.meshgrid(x, y, indexing="ij")
    x_pts = jnp.asarray([m.flatten() for m in m_grids]).T

    # Initial condition
    u0 = ic_field
    c = speed
    c2 = c**2

    # Wave numbers
    kx = fftfreq(N, d=dx) * 2 * jnp.pi
    ky = fftfreq(N, d=dx) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    K_squared = KX**2 + KY**2

    # Initial time stepping
    U0_hat = fft2(u0)
    Laplacian_u0_hat = -K_squared * U0_hat
    Laplacian_u0 = ifft2(Laplacian_u0_hat).real

    u1 = u0 + 0.5 * dt**2 * c2 * Laplacian_u0

    Nt = int(Tend / dt)

    # Initialize previous and current solutions
    u_nm1 = u0
    u_n = u1

    @jax.jit
    def time_step(u_nm1, u_n):
        U_hat = fft2(u_n)
        Laplacian_u_hat = -K_squared * U_hat
        Laplacian_u = ifft2(Laplacian_u_hat).real

        u_np1 = 2 * u_n - u_nm1 + dt**2 * c2 * Laplacian_u
        return u_n, u_np1  # Update previous and current solutions

    # Time-stepping using lax.scan
    def scan_fn(carry, _):
        u_nm1, u_n = carry
        u_nm1, u_n = time_step(u_nm1, u_n)
        return (u_nm1, u_n), u_n  # No outputs collected

    # Initial carry (u_nm1, u_n)
    init_carry = (u_nm1, u_n)

    # Number of steps to simulate (Nt - 1 because we already computed u1)
    num_steps = Nt - 1

    # Perform the time-stepping loop using lax.scan
    (_, _), sol = jax.lax.scan(scan_fn, init_carry, None, length=num_steps)

    sol = jnp.concatenate([u0[None], sol], axis=0)

    return sol


def get_wave_random_media(n_samples, t_pts, x_pts, key, batch_size=32, sigma=None):
    skey, sskey = jax.random.split(key)
    keys = jax.random.split(skey, num=n_samples)
    keys2 = jax.random.split(sskey, num=n_samples)

    grid = grids.Grid((x_pts, x_pts), domain=(
        (0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    max_velocity = 1

    if sigma is None:
        peak_wavenumber = 4
    else:
        peak_wavenumber = sigma

    def get_speed_field(key):
        v0 = cfd.initial_conditions.filtered_velocity_field(
            key, grid, max_velocity, peak_wavenumber
        )
        v0 = cfd.finite_differences.curl_2d(v0).data
        v0, _ = normalize(v0, method="01")
        return v0

    s_fields = vmap(get_speed_field)(keys)

    if sigma == 0:
        s_fields = np.ones_like(s_fields)*0.5

    def ic_fn(key):
        peak_wavenumber = 1
        v0 = cfd.initial_conditions.filtered_velocity_field(
            key, grid, max_velocity, peak_wavenumber
        )
        v0 = cfd.finite_differences.curl_2d(v0).data
        v0, _ = normalize(v0, method="01")
        return v0
    ic_fields = vmap(ic_fn)(keys2)

    Tend = 8.0
    dt = 4e-3

    @jit
    def solve(fields):
        ic_field, s_f = fields
        sol = solve_wave_equation(Tend, dt, x_pts,  ic_field, s_f)
        t_idx = jnp.linspace(0, len(sol) - 1, t_pts, dtype=jnp.int32)
        return sol[t_idx]

    fields = jnp.concatenate([ic_fields[:, None], s_fields[:, None]], axis=1)
    sols = batchvmap(solve, batch_size, in_arg=0)(fields)
    sols = np.asarray(sols)
    return sols
