
import numpy as np

import gmfm.io.result as R
from gmfm.config.config import Config
from gmfm.test.integrate import sample_model
from gmfm.test.metrics import compute_metrics
from gmfm.test.plot import plot_sde, plot_spde
from gmfm.utils.tools import jax_key_to_np, pshape, unnormalize


def run_test(cfg: Config, apply_fn, opt_params, x_data, key):

    plot = cfg.test.plot
    n_samples = cfg.test.n_samples

    # n_steps = cfg.integrate.n_steps
    sigma = cfg.loss.sigma

    print("sampling model...")
    rng = jax_key_to_np(key)
    n_idx = rng.integers(0, x_data.shape[0], size=n_samples)

    x_true = x_data[n_idx]
    x_0 = x_true[:, 0]
    T = x_true.shape[1]
    x_pred = sample_model(
        cfg, apply_fn, opt_params, x_0, sigma, T, key)

    x_pred = np.nan_to_num(
        x_pred, nan=0.0, posinf=1e9, neginf=-1e9)
    high_dim = x_true.ndim > 3

    pshape(x_true, x_pred)
    if plot:
        if high_dim:
            plot_spde(cfg, x_pred, x_true)
        else:
            plot_sde(cfg, x_pred, x_true)

    if cfg.test.metrics:

        stats = R.RESULT['normalize_values']
        x_pred = unnormalize(x_pred, stats)
        x_true = unnormalize(x_data, stats)
        pshape(x_true, x_pred)
        compute_metrics(cfg, x_pred, x_true)

    return x_pred
