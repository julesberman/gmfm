from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from gmfm.utils.tools import batchvmap, normalize


def solve_linear_advection_2D(ic_fn, c_x, c_y, N, dt, t_end):
    """
    Solve the 2D linear advection equation with time-dependent speeds:
        u_t + c_x(t) * u_x + c_y(t) * u_y = 0
    on the domain [0,1] x [0,1] with periodic boundaries, using
    a second-order MacCormack scheme. Speeds c_x, c_y are
    callables that take a scalar time t and return the speed at that time.
    Everything is JIT-compiled.

    Parameters
    ----------
    ic_fn : callable
        A JAX-compatible function ic_fn(X, Y) -> jnp.array of shape (N, N).
        X and Y each have shape (N, N) giving the coordinates of the grid.
        If you need to fix parameters in ic_fn, you can use functools.partial.
    N : int
        Number of grid points in both x and y directions.
    dt : float
        Time step size.
    t_end : float
        Final time of the simulation.
    c_x : callable
        A JAX-compatible function c_x(t) -> float. Partially applied if needed.
    c_y : callable
        A JAX-compatible function c_y(t) -> float. Partially applied if needed.

    Returns
    -------
    solution : jax.numpy.ndarray
        A 3D array of shape (T, N, N), where T is the total number
        of time steps (including t=0). solution[k, :, :] is the solution
        at time step k.

    Notes
    -----
    - The MacCormack scheme is second-order in time and space, assuming
      the solution is smooth and the CFL condition holds. For time-dependent
      speeds, we use c_x(t_n) in the predictor and c_x(t_{n+1}) in the corrector,
      similarly for c_y.
    - For stability, you'd generally want:
         |c_x(t)|*dt/dx + |c_y(t)|*dt/dy <= 1
      for all t in [0, t_end].
    """

    # Create spatial grid (periodic domain [0,1])
    x = jnp.linspace(0.0, 1.0, N, endpoint=False)
    y = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # 2D mesh
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # Number of time steps (including the initial condition)
    num_steps = jnp.asarray(jnp.floor(t_end / dt), dtype=jnp.int32) + 1

    # Initial condition
    u0 = ic_fn(X, Y)  # shape (N, N), JAX array
    t0 = 0.0

    # -------------------------------------------------------#
    # MacCormack step with time-dependent speeds.
    # We'll pass in c_xn = c_x(t_n), c_yn = c_y(t_n),
    # and c_xnp1 = c_x(t_{n+1}), c_ynp1 = c_y(t_{n+1}).
    # -------------------------------------------------------#
    def maccormack_step_time_dependent(u, c_xn, c_yn, c_xnp1, c_ynp1):
        # Predictor (forward difference) with c_xn, c_yn
        u_ip1 = jnp.roll(u, shift=-1, axis=0)  # i+1
        u_jp1 = jnp.roll(u, shift=-1, axis=1)  # j+1

        u_tilde = u \
                  - (c_xn * dt / dx) * (u_ip1 - u) \
                  - (c_yn * dt / dy) * (u_jp1 - u)

        # Corrector (backward difference) with c_xnp1, c_ynp1
        u_im1_tilde = jnp.roll(u_tilde, shift=1, axis=0)  # i-1
        u_jm1_tilde = jnp.roll(u_tilde, shift=1, axis=1)  # j-1

        u_next = 0.5 * (u + u_tilde) \
            - 0.5 * (c_xnp1 * dt / dx) * (u_tilde - u_im1_tilde) \
            - 0.5 * (c_ynp1 * dt / dy) * (u_tilde - u_jm1_tilde)

        return u_next

    # -------------------------------------------------------#
    # We define a "time_step" function to be used in lax.scan.
    # It advances (u, t) -> (u_next, t_next).
    # -------------------------------------------------------#
    def time_step(carry, _):
        u_current, t_current = carry

        # Speeds at t_n and t_{n+1}
        cx_n = c_x(t_current)
        cy_n = c_y(t_current)
        cx_np1 = c_x(t_current + dt)
        cy_np1 = c_y(t_current + dt)

        # Next solution
        u_next = maccormack_step_time_dependent(u_current, cx_n, cy_n,
                                                cx_np1, cy_np1)
        t_next = t_current + dt
        return (u_next, t_next), u_next

    # We'll run the simulation for (num_steps-1) iterations
    # because we already have the initial condition at t=0.
    scan_init = (u0, t0)
    _, states = jax.lax.scan(time_step, scan_init, jnp.arange(num_steps - 1))

    # states has shape (num_steps-1, N, N). Prepend the initial condition:
    solution = jnp.concatenate([u0[jnp.newaxis, ...], states], axis=0)
    # shape: (num_steps, N, N)

    return solution


def sample_rbf_gp(key,
                  n_samples,
                  x_grid,
                  sigma: float = 1.0,
                  lengthscale: float = 0.2,
                  jitter: float = 4e-6):

    n_x = len(x_grid)

    # 2) Build the RBF covariance matrix
    x_col = x_grid[:, None]
    x_row = x_grid[None, :]
    dist_sq = (x_col - x_row) ** 2
    K_g = sigma**2 * jnp.exp(-0.5 * dist_sq / (lengthscale**2))

    # 3) Add jitter to the diagonal to ensure numerical stability
    K_g += jitter * jnp.eye(n_x)

    # 4) Cholesky decomposition
    L = jax.scipy.linalg.cholesky(K_g).T  # shape (n_x, n_x)

    # 5) Sample standard normals (shape: (n_x, n_samples)) & transform
    eps = jax.random.normal(key, shape=(n_x, n_samples))

    # multiply by L to get correlated samples, then transpose
    gp_samples = (L @ eps).T  # shape (n_samples, n_x)

    return gp_samples


def get_K_adv(n_samples, K, key, sigma=1, batch_size=32, rand_ic=True, t_end=0.5):
    def get_stochastic_adv(key):
        # Simulation parameters
        N = 256
        dt = 5e-4
        key, ickey = jax.random.split(key)
        roll = jax.random.choice(ickey, N-1, shape=())

        def ic_fn(X, Y):
            Y -= 0.25
            X -= 0.5
            v = 200.0
            f = jnp.exp(-v * ((X)**2 + (Y)**2))
            if rand_ic:
                f = jnp.roll(f, roll, axis=0)
            return f

        x_space = jnp.linspace(0, t_end, 128)
        f = sample_rbf_gp(key, 1, x_space, sigma=sigma, lengthscale=0.1)[0]
        f = normalize(f, '-11')[0]

        def cx_fn(t):
            t = jnp.asarray([t])
            return jnp.interp(t, x_space, f)*K

        def cy_fn(t):
            return 1.0

        # Solve the advection equation
        solve_fn = partial(solve_linear_advection_2D, ic_fn, cx_fn, cy_fn)
        sol = solve_fn(N, dt, t_end)
        ss_x = 2
        ss_t = 10
        sol = sol[::ss_t, ::ss_x, ::ss_x]
        return sol

    keys = jax.random.split(key, num=n_samples)

    sols = batchvmap(get_stochastic_adv, batch_size, in_arg=0)(keys)

    sols = np.asarray(sols)
    return sols


def get_adv_data(n_samples, sub_t, sub_x):

    dir = Path("/scratch/jmb1174/sadv/rand_ic_sols.h5")

    with h5py.File(dir, "r") as h5_file:
        sols = h5_file["videos"][:n_samples, ::sub_t, ::sub_x, ::sub_x]

    print(sols.shape, sols.dtype)

    return sols
