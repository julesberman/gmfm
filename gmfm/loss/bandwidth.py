import jax
import jax.numpy as jnp


def median_heuristic_sigma(x, key, n_subsample=1024, eps=1e-8):
    """
    x: (N, D) samples at a fixed time
    Returns: scalar sigma = median pairwise Euclidean distance (subsampled)
    """
    N, D = x.shape
    n = jnp.minimum(n_subsample, N)

    idx = jax.random.choice(key, N, shape=(n,), replace=False)
    xs = x[idx]  # (n, D)

    # Pairwise distances (n, n)
    diffs = xs[:, None, :] - xs[None, :, :]
    d2 = jnp.sum(diffs * diffs, axis=-1)
    d = jnp.sqrt(d2 + eps)

    # Take upper triangle without diagonal
    iu = jnp.triu_indices(n, k=1)
    du = d[iu]

    sigma = jnp.median(du)
    return jnp.maximum(sigma, eps)
