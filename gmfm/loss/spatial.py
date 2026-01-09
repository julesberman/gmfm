import jax
import jax.numpy as jnp
import numpy as np
from jax import jit


# -----------------------------
# Spatial Fourier features on grid
# -----------------------------
def make_grid_coords_unit(H, W):
    """
    Pixel-center coords in [0,1) x [0,1).
    Returns coords: (HW, 2).
    """
    xs = (jnp.arange(W) + 0.5) / W
    ys = (jnp.arange(H) + 0.5) / H
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")  # (H,W)
    coords = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=1)  # (HW,2)
    return coords


def sample_discrete_spatial_modes_np(rng, M, H, W, sigmas, avoid_zero=True):
    """
    Sample integer Fourier modes (kx, ky) as a mixture over bandwidths (sigmas),
    then map to angular frequencies for coords in [0,1): kappa = 2π [kx, ky].

    Analogue to RFF omega ~ N(0, sigma^{-2} I):
      kappa ~ N(0, sigma^{-2} I_2)
    but with kappa = 2π k_int, so
      k_int ~ N(0, (2π sigma)^{-2} I_2).

    Parameters
    ----------
    rng : np.random.Generator
    M : int
        number of modes
    H, W : int
        grid size
    sigmas : float or array-like
        bandwidth(s); smaller sigma => higher frequencies
    avoid_zero : bool
        resample any (kx,ky)=(0,0)

    Returns
    -------
    kappa : np.ndarray, shape (M, 2), dtype float32
    """
    sigmas = np.atleast_1d(np.asarray(sigmas, dtype=float))
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be a scalar or 1D array-like.")

    L = sigmas.shape[0]

    # split M across sigmas (same pattern as your make_rff_params)
    M_base = M // L
    rem = M - M_base * L
    counts = M_base + (np.arange(L) < rem).astype(np.int32)  # sums to M

    # valid integer mode ranges for a W-point / H-point periodic grid
    kx_min, kx_max = -(W // 2), (W // 2) - 1
    ky_min, ky_max = -(H // 2), (H // 2) - 1

    # sample per-sigma blocks
    kx_blocks = []
    ky_blocks = []

    for sigma, m in zip(sigmas, counts):
        if m == 0:
            continue

        # std of integer mode corresponding to kappa ~ N(0, sigma^{-2} I_2)
        std_k = 1.0 / (2.0 * np.pi * sigma)

        # guard against degeneracy for very large sigma (otherwise almost all k become 0)
        std_k = max(std_k, 1.0)

        kx = np.rint(rng.normal(loc=0.0, scale=std_k, size=m)).astype(np.int32)
        ky = np.rint(rng.normal(loc=0.0, scale=std_k, size=m)).astype(np.int32)

        kx = np.clip(kx, kx_min, kx_max)
        ky = np.clip(ky, ky_min, ky_max)

        kx_blocks.append(kx)
        ky_blocks.append(ky)

    kx = np.concatenate(kx_blocks, axis=0)
    ky = np.concatenate(ky_blocks, axis=0)

    # avoid the zero mode if requested by resampling those entries from the same mixture
    if avoid_zero:
        probs = counts.astype(float) / max(1, counts.sum())

        mask = (kx == 0) & (ky == 0)
        while np.any(mask):
            idx = np.where(mask)[0]
            # choose which sigma component each resampled index comes from
            comp = rng.choice(L, size=idx.size, p=probs)

            # draw new k's for these indices
            # (vectorized by building per-index std_k)
            std_k = 1.0 / (2.0 * np.pi * sigmas[comp])
            std_k = np.maximum(std_k, 1.0)

            kx_new = np.rint(rng.normal(0.0, std_k)).astype(np.int32)
            ky_new = np.rint(rng.normal(0.0, std_k)).astype(np.int32)

            kx_new = np.clip(kx_new, kx_min, kx_max)
            ky_new = np.clip(ky_new, ky_min, ky_max)

            kx[idx] = kx_new
            ky[idx] = ky_new

            mask = (kx == 0) & (ky == 0)

    kappa = np.stack([2.0 * np.pi * kx, 2.0 * np.pi * ky],
                     axis=1).astype(np.float32)
    return kappa


def make_spatial_feature_matrix(
    key,
    H: int,
    W: int,
    C: int,
    M: int,
    sigmas: list,
    use_bias: bool = True,
    normalize: bool = True,
):
    """
    Builds Psi of shape (D, 2M) where D=H*W*C (channels last flattening).
    Psi columns are cos(kappa^T s + b) and sin(kappa^T s + b) evaluated on grid coords s.
    Replicates the same spatial features across channels.

    Returns:
      Psi: (D, 2M) jnp.ndarray
      kappa: (M,2) np.ndarray (for diagnostics)
    """
    coords = make_grid_coords_unit(H, W)  # (HW,2)

    # sample kappa in NumPy for reproducibility with your existing style
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    kappa = sample_discrete_spatial_modes_np(rng, M, H, W, sigmas)

    kappa_j = jnp.asarray(kappa)         # (M,2)
    z = coords @ kappa_j.T               # (HW,M)

    if use_bias:
        b = jax.random.uniform(key, shape=(M,), minval=0.0, maxval=2 * jnp.pi)
        z = z + b[None, :]

    Psi_hw = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=1)  # (HW,2M)

    if normalize:
        Psi_hw = Psi_hw / jnp.sqrt(M)

    # channels-last flattening: (H,W,C) -> (HWC)
    # repeat each pixel's feature C times so each channel at that pixel uses same spatial basis
    Psi = jnp.repeat(Psi_hw, repeats=C, axis=0)  # (HW*C, 2M)

    Psi = jnp.swapaxes(Psi, 0, 1)
    return Psi, kappa


@jit
def spatial_phi(u, Psi):
    """
    u:   (B,D) flattened fields (H*W*C)
    Psi: (2M, D)
    returns: (2M,) empirical mean of <u, psi_k> over batch
    """
    feats = u @ Psi.T  # (B,2M)
    return jnp.mean(feats, axis=0)


@jit
def spatial_grad_dot_v(x_t, v_t, Psi):
    """
    Psi: (2M, D)
    v: (B,D)
    returns: (2M,) empirical mean of <v, psi_k> over batch
    """
    feats = v_t @ Psi.T  # (B,2M)
    return jnp.mean(feats, axis=0)


@jit
def spatial_laplace_phi(u, Psi):
    """
    Laplacian w.r.t. state u: Δ_u <u,psi> = 0.
    Kept for API compatibility.
    """
    return jnp.zeros((Psi.shape[0],), dtype=u.dtype)
