
import numpy as np

import gmfm.io.result as R
from gmfm.config.config import Config
from gmfm.test.integrate import sample_model
from gmfm.test.metrics import compute_metrics
from gmfm.test.plot import plot_sde, plot_spde
from gmfm.utils.tools import jax_key_to_np, pshape, unnormalize

import jax.numpy as jnp


def run_test(cfg: Config, apply_fn, opt_params, x_data, cur_mu, key, label=''):

    plot = cfg.test.plot
    n_samples = cfg.test.n_samples

    # n_steps = cfg.integrate.n_steps
    sigma = cfg.loss.sigma

    print("sampling model...")
    rng = jax_key_to_np(key)

    if n_samples == -1:
        x_true = x_data
    else:
        n_idx = rng.choice(x_data.shape[0], size=n_samples, replace=False)
        x_true = x_data[n_idx]

    if cfg.test.t_samples is not None:

        t_pts = min(cfg.test.t_samples, x_true.shape[1])
        t_idx = np.linspace(0,  x_true.shape[1] - 1, t_pts,
                            endpoint=True, dtype=np.int32)
        x_true = x_true[:, t_idx]

    if cfg.data.has_mu:
        def apply_fn_mu(params, xt, t):
            mu = jnp.ones_like(t)*cur_mu
            return apply_fn(params, xt, t, mu)
    else:
        def apply_fn_mu(params, xt, t):
            return apply_fn(params, xt, t, None)

    T = x_true.shape[1]
    t_int = jnp.linspace(0.0, 1.0, T)

    # if skip ic
    # t_int = t_int[1:]
    x_0 = x_true[:, 0]
    x_pred = sample_model(
        cfg, apply_fn_mu, opt_params, x_0, sigma, t_int, key)

    x_pred = np.nan_to_num(
        x_pred, nan=0.0, posinf=1e9, neginf=-1e9)
    high_dim = x_true.ndim > 3

    if plot:
        if high_dim:
            plot_spde(cfg, x_pred, x_true)
        else:
            plot_sde(cfg, x_pred, x_true, label=label)

    if cfg.test.metrics:

        stats = R.RESULT['normalize_values']
        x_pred = unnormalize(x_pred, stats)
        x_true = unnormalize(x_true, stats)
        x_pred = jnp.squeeze(x_pred)
        x_true = jnp.squeeze(x_true)
        pshape(x_true, x_pred)
        compute_metrics(cfg, x_pred, x_true, label=label)

    return x_pred
