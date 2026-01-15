from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from interpax import approx_df
from jax import jit, lax, vmap
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm

import gmfm.io.result as R
from gmfm.config.config import Config
from gmfm.loss.bandwidth import median_heuristic_sigma
from gmfm.utils.tools import (
    batchvmap,
    jax_key_to_np,
    print_ndarray,
    print_stats, 
    pshape,
)


def get_phi_params(cfg: Config, x_data, t_data, key):

    lhs_data = []

    x_orig = x_data
    x_flat = rearrange(x_orig, "N T ... -> N T (...)")
    N = x_orig.shape[0]
    T = x_orig.shape[1]

    k1, k2, k3 = jax.random.split(key, num=3)


    omega_rho = cfg.loss.omega_rho
    n_functions = cfg.loss.n_functions
    kpn = jax_key_to_np(k1)
    stride = int(cfg.loss.stride)

    # sigma_t = []
    # t_flat_data = rearrange(x_flat, 'N T D -> T N D')
    # for xt in t_flat_data:
    #     kk, k2 = jax.random.split(k2)
    #     ss = median_heuristic_sigma(xt, kk)
    #     sigma_t.append(ss)
    # sigma_t = np.asarray(sigma_t)
    # print_ndarray(sigma_t)

    if cfg.loss.b_min > 0:
        bandwidths = np.exp(np.linspace(np.log(cfg.loss.b_min), np.log(cfg.loss.b_max), cfg.loss.n_bands))
    else: 
        bandwidths = np.asarray(list(cfg.loss.bandwidths))
    print_ndarray(bandwidths)

    R.RESULT['bandwidths'] = bandwidths

    _, _, D = x_flat.shape
    kpn = jax_key_to_np(k1)

    if omega_rho == 'gauss':
        fixed_params = make_rff_params(
            kpn, n_functions, D, bandwidths)  # omega: (M,D)
    if omega_rho == 'orf':
        fixed_params = make_rff_params_orf(
            kpn, n_functions, D, bandwidths)  # omega: (M,D)
    # ---- precompute mu_t for all times: mu[t] = E[phi(X_t)] ----
    pbar_mu = tqdm(range(T), desc='precompute moments', colour="#306EFF")
    mu_list = []
    for t_idx in pbar_mu:
        xt = x_flat[:, t_idx]  # (N,D)
        if N > 25_000:
            mu_t = rff_phi_chunked(xt, fixed_params)  # (2M,)
        else:
            mu_t = rff_phi(xt, fixed_params)          # (2M,)

        mu_list.append(np.asarray(mu_t))

    mu = np.stack(mu_list, axis=0)  # (T, 2M)

    dt_method = cfg.loss.dt

    if dt_method == 'sm_spline':
        spl = make_smoothing_spline(t_data, mu, lam=5e-5)
        lhs_data = spl.derivative()(t_data)      
    elif dt_method == 'sm_fd':
        mu_s = gaussian_filter1d(mu, sigma=2.0, axis=0, mode="nearest")     
        lhs_data = np.gradient(mu_s, t_data, axis=0, edge_order=2)
    elif dt_method == 'fd':
        lhs_data = get_dt_finite_difference(mu, t_data, stride)
    else:
        lhs_data = get_dt_spline(mu, t_data, dt_method)

    pshape(lhs_data, fixed_params)
    print_stats(lhs_data)
    print_stats(fixed_params)

    return np.asarray(mu), np.asarray(lhs_data), np.asarray(fixed_params), bandwidths


def get_dt_spline(mu_data, t_data, method):

    @jit
    def get_dt_mu(mu_t):
        return approx_df(t_data, mu_t, method=method)

    dt_mu = vmap(get_dt_mu, in_axes=1, out_axes=1)(mu_data)

    return np.asarray(dt_mu)


def get_dt_finite_difference(mu, t_data, stride):
    # ---- build lhs_data with T entries using stride-aware stencils ----
    # Convention: lhs[t] approximates (2*stride*dt) * d/dt E[phi(X_t)]
    # so interior central: lhs[t] = mu[t+stride] - mu[t-stride]
    # left one-sided (2nd order): lhs[t] = -3 mu[t] + 4 mu[t+stride] - mu[t+2stride]
    # right one-sided (2nd order): lhs[t] =  3 mu[t] - 4 mu[t-stride] + mu[t-2stride]
    # (fallback to 1st order one-sided if needed)
    lhs_list = []
    dt = t_data[stride] - t_data[0]
    T = mu.shape[0]
    for t_idx in range(T):
        has_central = (t_idx - stride >= 0) and (t_idx + stride < T)
        has_fwd1 = (t_idx + stride < T)
        has_bwd1 = (t_idx - stride >= 0)

        if has_central:
            lhs = (mu[t_idx + stride] - mu[t_idx - stride]) / (2.0 * dt)
        elif has_fwd1:
            lhs = (mu[t_idx + stride] - mu[t_idx]) / dt
        elif has_bwd1:
            lhs = (mu[t_idx] - mu[t_idx - stride]) / dt

        lhs_list.append(lhs)

    lhs_data = np.stack(lhs_list, axis=0)  # (T, 2M)

    return lhs_data


def make_rff_params(
    key,
    M_total: int,
    D: int,
    sigmas,
):
    """
    Pure NumPy version of make_rff_params.

    Parameters
    ----------
    key : int | np.random.Generator
        RNG seed (int) or an existing NumPy Generator.
    M_total : int
        Total number of random frequencies.
    D : int
        Input dimension.
    sigmas : float or array-like of shape (L,)
        If array, splits M_total approximately evenly across sigmas.

    Returns
    -------
    omega : np.ndarray of shape (M_total, D)
    """
    rng = key if isinstance(
        key, np.random.Generator) else np.random.default_rng(key)

    sigmas = np.atleast_1d(np.asarray(sigmas, dtype=np.float32))
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be a scalar or 1D array-like.")

    L = sigmas.shape[0]

    M_base = M_total // L
    rem = M_total - M_base * L  # same as M_total % L
    # (L,), sums to M_total
    counts = M_base + (np.arange(L) < rem).astype(np.int32)

    base = rng.standard_normal((M_total, D), dtype=np.float32)

    sigma_per = np.repeat(sigmas, repeats=counts)  # (M_total,)
    omega = base / sigma_per[:, None]              # (M_total, D)

    return omega


@partial(jax.jit, static_argnames=("M_total", "D"))
def make_rff_params_jnp(key, M_total: int, D: int, sigma):
    base = jax.random.normal(key, (M_total, D))  # (M_use, D)
    omega = base / sigma
    return omega


def rff_phi_chunked(x, omega):
    """
    x: (B, D), omega: (M, D)
    returns: (B, 2M) with [cos, sin]
    """
    scale = 1.0

    @jit
    def single(x):
        """
        x: (D), omega: (M, D)
        returns: (2M) with [cos, sin]
        """
        z = x @ omega.T
        return jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)

    res = batchvmap(single, batch_size=25_000, pbar=False,
                    to_numpy=True, mean=True)(x)

    return res*scale


@jit
def rff_phi(x, omega):
    """
    x: (B, D), omega: (M, D)
    returns: (2M,) with [E cos, E sin] (mean over batch)
    """
    z = x @ omega.T  # (B, M)
    scale = 1.0  # keep consistent with your current scaling

    cos_mean = jnp.mean(jnp.cos(z), axis=0)  # (M,)
    sin_mean = jnp.mean(jnp.sin(z), axis=0)  # (M,)

    return scale * jnp.concatenate([cos_mean, sin_mean], axis=0)  # (2M,)


@jit
def rff_grad_dot_v(x, v, omega):
    """
    returns (B, 2M) with [∇cos·v, ∇sin·v]
    where:
      ∇cos(ωᵀx)·v = -sin(ωᵀx) * (ωᵀv)
      ∇sin(ωᵀx)·v =  cos(ωᵀx) * (ωᵀv)

    x:     (B, D)
    v:     (B, D)
    omega: (M, D)

    Returns:
      (2M,) = concat([mean(-sin(z) * (v@omega.T), axis=0),
                      mean( cos(z) * (v@omega.T), axis=0)])
    """
    # (B, M)
    z = x @ omega.T
    omega_dot = v @ omega.T

    # Reduce each half first: (M,), then concat to (2M,)
    left = jnp.mean(-jnp.sin(z) * omega_dot, axis=0)  # (M,)
    right = jnp.mean(jnp.cos(z) * omega_dot, axis=0)  # (M,)

    return jnp.concatenate([left, right], axis=0)


@jit
def rff_laplace_phi(x, omega):
    """
    x: (B, D), omega: (M, D)
    returns: (2M,) with [E Δcos, E Δsin] (mean over batch)

    Δcos(ωᵀx) = -||ω||^2 cos(ωᵀx)
    Δsin(ωᵀx) = -||ω||^2 sin(ωᵀx)
    """
    z = x @ omega.T                              # (B, M)
    w2 = jnp.sum(omega * omega, axis=-1)         # (M,)
    scale = 1.0  # keep consistent with your current scaling

    # Multiply by w2 via broadcasting, then reduce to (M,) directly.
    lap_cos_mean = jnp.mean(-jnp.cos(z) * w2[None, :], axis=0)  # (M,)
    lap_sin_mean = jnp.mean(-jnp.sin(z) * w2[None, :], axis=0)  # (M,)

    # (2M,)
    return scale * jnp.concatenate([lap_cos_mean, lap_sin_mean], axis=0)


@jax.jit(static_argnames=("chunk",))
def rff_grad_dot_v_chunked(x, v, omega, *, chunk: int = 16384):
    B = x.shape[0]
    M = omega.shape[0]
    pad = (-M) % chunk  # now concrete because chunk is static

    omega_p = jnp.pad(omega, ((0, pad), (0, 0)))  # OK: pad is concrete
    Mp = omega_p.shape[0]
    n = Mp // chunk

    out = jnp.zeros((2 * Mp,), dtype=x.dtype)

    def body(i, out):
        lo = i * chunk
        om = lax.dynamic_slice_in_dim(omega_p, lo, chunk, axis=0)  # (chunk, D)

        z = x @ om.T
        od = v @ om.T

        a0 = jnp.sum(-jnp.sin(z) * od, axis=0) / B
        a1 = jnp.sum(jnp.cos(z) * od, axis=0) / B

        out = lax.dynamic_update_slice(out, a0, (lo,))
        out = lax.dynamic_update_slice(out, a1, (Mp + lo,))
        return out

    out = lax.fori_loop(0, n, body, out)
    return out[: 2 * M]


@jax.jit(static_argnames=("chunk",))
def rff_laplace_phi_chunked(x, omega, *, chunk: int = 16384):
    B = x.shape[0]
    M = omega.shape[0]
    pad = (-M) % chunk

    omega_p = jnp.pad(omega, ((0, pad), (0, 0)))
    Mp = omega_p.shape[0]
    n = Mp // chunk

    out = jnp.zeros((2 * Mp,), dtype=x.dtype)

    def body(i, out):
        lo = i * chunk
        om = lax.dynamic_slice_in_dim(omega_p, lo, chunk, axis=0)

        z = x @ om.T
        w2 = jnp.sum(om * om, axis=-1)  # (chunk,)

        a0 = jnp.sum((-jnp.cos(z)) * w2[None, :], axis=0) / B
        a1 = jnp.sum((-jnp.sin(z)) * w2[None, :], axis=0) / B

        out = lax.dynamic_update_slice(out, a0, (lo,))
        out = lax.dynamic_update_slice(out, a1, (Mp + lo,))
        return out

    out = lax.fori_loop(0, n, body, out)
    return out[: 2 * M]


def grad_phi_weights(x_t, omegas, eps=1e-8):
    """
    Returns weights w of shape (2M,) where
      w_k = 1 / (E[||∇phi_k||^2] + eps)
    for paired tests phi_cos, phi_sin.
    """
    # (B, M): dot products ω^T x
    proj = x_t @ omegas.T

    # (B, M)
    sinp = jnp.sin(proj)
    cosp = jnp.cos(proj)

    # (M,)
    omega_norm2 = jnp.sum(omegas**2, axis=-1)

    # E[||∇cos||^2] = E[sin^2(proj)] * ||ω||^2
    # E[||∇sin||^2] = E[cos^2(proj)] * ||ω||^2
    s2_cos = jnp.mean(sinp**2, axis=0) * omega_norm2  # (M,)
    s2_sin = jnp.mean(cosp**2, axis=0) * omega_norm2  # (M,)

    # Stack to match your (2M,) feature order.
    # If your features are [cos_1..cos_M, sin_1..sin_M], do:
    s2 = jnp.concatenate([s2_cos, s2_sin], axis=0)    # (2M,)

    w = 1.0 / (s2 + eps)
    return w


def make_rff_params_orf(
    key,
    M_total: int,
    D: int,
    sigmas,
):
    """
    JAX ORF version of make_rff_params (same API, supports multiple bandwidths),
    returning a NumPy array at the end.

    ORF block construction (rows are frequencies):
        W = (1/sigma) * S Q
    where:
        Q ~ Haar(O(D))  (via jax.random.orthogonal)
        S = diag(s_1,...,s_D), s_i ~ chi_D i.i.d.  (row-wise scaling)
    """

    # ---- parse sigmas & split counts exactly like your NumPy version ----
    sigmas = np.atleast_1d(np.asarray(sigmas, dtype=np.float32))
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be a scalar or 1D array-like.")
    L = int(sigmas.shape[0])

    M_base = M_total // L
    rem = M_total - M_base * L
    counts = (M_base + (np.arange(L) < rem).astype(np.int32)
              ).astype(int)  # sums to M_total

    @partial(jax.jit, static_argnames=("M",))
    def _orf_for_sigma(k, M: int, sigma: float) -> jax.Array:
        """Generate (M, D) ORF frequencies for a single sigma."""
        if M == 0:
            return jnp.empty((0, D), dtype=jnp.float32)

        B = (M + D - 1) // D  # number of D-row blocks

        kQ, kZ = jax.random.split(k, 2)

        # Q: (B, D, D) Haar-orthogonal blocks
        Q = jax.random.orthogonal(kQ, n=D, shape=(B,), dtype=jnp.float32)

        # s_i ~ chi_D: s = sqrt(sum_{k=1..D} z_k^2), z_k ~ N(0,1)
        Z = jax.random.normal(kZ, shape=(B, D, D), dtype=jnp.float32)
        s = jnp.sqrt(jnp.sum(Z * Z, axis=-1))  # (B, D)

        # Row-scale Q by s and divide by sigma: W[b, i, :] = s[b, i] * Q[b, i, :] / sigma
        W = (Q * s[..., :, None]) / jnp.float32(sigma)  # (B, D, D)

        # Flatten blocks into rows and truncate to M
        return W.reshape((B * D, D))[:M]

    # ---- generate each sigma-group with independent keys, then concatenate ----
    keys = jax.random.split(key, L) if L > 0 else jnp.empty(
        (0,), dtype=key.dtype)

    parts = []
    for k_l, sigma_l, cnt_l in zip(keys, sigmas, counts):
        if cnt_l > 0:
            parts.append(_orf_for_sigma(k_l, int(cnt_l), float(sigma_l)))

    omega_jax = jnp.concatenate(parts, axis=0) if parts else jnp.empty(
        (0, D), dtype=jnp.float32)

    # ---- return NumPy on host ----
    return np.asarray(jax.device_get(omega_jax), dtype=np.float32)
