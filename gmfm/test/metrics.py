import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange

import gmfm.io.result as R
from gmfm.config.config import Config, get_outpath


def compute_metrics(cfg: Config, x_pred, x_true):

    outdir = get_outpath()
    compute_mean_relerr(x_pred, x_true)

    if cfg.dataset == 'turb':
        plot_enstrophy(x_pred, x_true, save_plots=True, outdir=outdir)


def compute_mean_relerr(pred, true):

    p_mean, t_mean = pred.mean(axis=0), true.mean(axis=0)
    p_mean = rearrange(p_mean, "T ... -> T (...)")
    t_mean = rearrange(t_mean, "T ... -> T (...)")
    err = jnp.linalg.norm(p_mean - t_mean) / jnp.linalg.norm(t_mean)
    err_time = jnp.linalg.norm(p_mean - t_mean, axis=-1) / jnp.linalg.norm(
        t_mean, axis=-1
    )

    R.RESULT["mean_err_time"] = err_time
    R.RESULT["mean_err"] = err

    print(f"mean rel_err: {err:.5f}")

    return err, err_time


def plot_enstrophy(
    pred_vort,  # (N,T,H,W,C)
    true_vort,  # (N,T,H,W,C)
    *,
    times=None,            # (T,) optional
    figsize=(6, 4),
    save_plots=False,
    outdir=".",
    label="enstrophy",
):
    """
    Single plot: ensemble mean enstrophy ± 1σ for true vs pred.
    Enstrophy per sample: Z[n,t] = mean_{x,y}( 0.5 * sum_c omega^2 ).
    """
    pred_vort = np.asarray(pred_vort)
    true_vort = np.asarray(true_vort)

    N, T, H, W, C = pred_vort.shape
    t = np.asarray(times) if times is not None else np.arange(T)

    def enstrophy_per(vort):
        e = 0.5 * np.sum(vort**2, axis=-1)          # (N,T,H,W)
        return e.mean(axis=(-2, -1))               # (N,T)

    Z_true = enstrophy_per(true_vort)
    Z_pred = enstrophy_per(pred_vort)

    Zt_m, Zt_s = Z_true.mean(axis=0), Z_true.std(axis=0)
    Zp_m, Zp_s = Z_pred.mean(axis=0), Z_pred.std(axis=0)

    R.RESULT['Zt_m'] = Zt_m
    R.RESULT['Zt_s'] = Zt_s
    R.RESULT['Zp_m'] = Zp_m
    R.RESULT['Zp_s'] = Zp_s

    plt.figure(figsize=figsize)
    plt.plot(t, Zt_m, label="true mean")
    plt.fill_between(t, Zt_m - Zt_s, Zt_m + Zt_s, alpha=0.2, label="true ±1σ")
    plt.plot(t, Zp_m, label="pred mean")
    plt.fill_between(t, Zp_m - Zp_s, Zp_m + Zp_s, alpha=0.2, label="pred ±1σ")
    plt.xlabel("time")
    plt.ylabel("enstrophy")
    plt.title("Enstrophy (ensemble)")
    plt.legend()
    plt.tight_layout()

    if save_plots:
        os.makedirs(outdir, exist_ok=True)
        plt.gcf().savefig(
            f"{outdir}/hit_{label}.png",
        )
        plt.cla()
        plt.clf()
    else:
        plt.show()
