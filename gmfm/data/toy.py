import math

import jax.numpy as jnp
import numpy as np

from gmfm.utils.tools import normalize


def _sample_swiss_roll(n, rng=None):
    rng = np.random.default_rng(rng)
    t = rng.uniform(1.5 * math.pi, 4.5 * math.pi, size=n)
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.stack([x, y], axis=1)
    return data * 0.1


def _sample_checkerboard(n, rng=None):
    rng = np.random.default_rng(rng)
    i = rng.integers(0, 4, size=n)
    j = rng.integers(0, 4, size=n)
    i = np.where(((i + j) & 1) == 0, i, (i + 1) & 3)  # keep every other cell
    u = rng.random((n, 2))
    x = -1.0 + (i + u[:, 0]) * 0.5
    y = -1.0 + (j + u[:, 1]) * 0.5
    # 0..7 (row-major over kept squares)
    cls = 2 * j + (i >> 1)
    return np.c_[x, y], cls


def _sample_multimodal_gaussian(n, rng=None, num_modes=6, radius=1.0, sigma=0.1):
    rng = np.random.default_rng(rng)
    angles = np.linspace(0, 2 * math.pi, num_modes, endpoint=False)
    means = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    m = (n + num_modes - 1) // num_modes               # points per mode (ceil)
    cls = np.repeat(np.arange(num_modes), m)
    x = means[cls] + rng.normal(0, sigma, size=(cls.size, 2))
    p = rng.permutation(cls.size)
    return x[p], cls[p]


def _sample_checkerboard_3(n, rng=None):
    rng = np.random.default_rng(rng)
    kept = np.array([(i, j) for j in range(3)
                    for i in range(3) if ((i + j) & 1) == 0], int)  # (5,2)
    K = len(kept)
    # points per kept cell (ceil)
    m = (n + K - 1) // K
    cls = np.repeat(np.arange(K), m)
    ij = kept[cls]
    u = rng.random((cls.size, 2))
    s = 2.0 / 3.0
    xy = -1.0 + (ij + u) * s
    p = rng.permutation(cls.size)
    return xy[p], cls[p]


def get_2d_dataset(name: str, n_samples: int = 50_000, seed: int | None = None):
    rng = np.random.default_rng(seed)
    name = name.lower()

    if "swiss" in name:
        data = _sample_swiss_roll(n_samples, rng)
    elif "checker" in name:
        data = _sample_checkerboard(n_samples, rng)
    elif "gmm" in name:
        data = _sample_multimodal_gaussian(n_samples, rng)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # data = _normalize_to_unit_square(data)
    data, _ = normalize(data, method='-11')
    data = jnp.asarray(data)

    return data, (2,)
