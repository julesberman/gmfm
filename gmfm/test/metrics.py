import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange

import gmfm.io.result as R
from gmfm.config.config import Config, get_outpath
import scipy.sparse as sp
from einops import rearrange
from jax import random as jrandom
from scipy.sparse.linalg import spsolve

from ott import utils
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from tqdm import tqdm
from jax import jit


def compute_metrics(cfg: Config, x_pred, x_true, label):

    outdir = get_outpath()
    compute_mean_relerr(x_pred, x_true)

    if cfg.dataset == 'turb':
        plot_enstrophy(x_pred, x_true, save_plots=True, outdir=outdir)

    if cfg.dataset in ['vtwo', 'vbump', 'v6']:
        x_true = np.swapaxes(x_true, 0, 1)
        x_pred = np.swapaxes(x_pred, 0, 1)
        true_sol_ele = x_true
        test_sol_ele = x_pred
        boxsize = 50

        if true_sol_ele.shape[-1] > 2:
            true_sol_ele = np.stack(
                [x_true[:, :, 0], x_true[:, :, 3]], axis=-1)
            test_sol_ele = np.stack(
                [x_pred[:, :, 0], x_pred[:, :, 3]], axis=-1)
            boxsize = 4 * np.pi

        true_electric = compute_electric_energy(true_sol_ele, boxsize=boxsize)
        test_electric = compute_electric_energy(test_sol_ele, boxsize=boxsize)
        R.RESULT[f"true_electric_{label}"] = true_electric
        R.RESULT[f"test_electric_{label}"] = test_electric

        outdir = get_outpath()
        t_int = np.linspace(0, 1, len(test_electric))
        plt.semilogy(t_int, true_electric, label="True")
        plt.semilogy(t_int, test_electric, label="Test")
        plt.xlabel("time")
        plt.ylabel("electric energy")
        plt.savefig(f"{outdir}/electric_{label}.png")
        plt.clf()
        err_electric = np.abs(
            true_electric - test_electric) / np.abs(true_electric)
        err_electric = np.mean(err_electric)
        R.RESULT[f"err_electric_{label}"] = err_electric
        print(f"err_electric_{label}: {err_electric:.3e}")

        if 'all_err_electric' not in R.RESULT:
            R.RESULT['all_err_electric'] = [err_electric]
        else:
            R.RESULT['all_err_electric'].append(err_electric)

        print(f"computing wasserstein")

        epsilon = 1e-3
        n_wass_time = 16

        t_idx = np.linspace(0, 1, n_wass_time, dtype=np.int32)

        n_sample_wass = 5_000

        test_sol_wass = x_true[t_idx, :n_sample_wass]
        true_sol_wass = x_pred[t_idx, :n_sample_wass]

        w_time = compute_wasserstein_time(
            test_sol_wass, true_sol_wass, eps=epsilon)

        R.RESULT[f"time_wass_dist_{label}"] = w_time
        mean_w_dist = np.mean(w_time)
        R.RESULT[f"mean_wass_dist_{label}"] = mean_w_dist
        print(f"mean_wass_dist_{label}: {mean_w_dist:.3e}")

        if 'all_wass_dist' not in R.RESULT:
            R.RESULT['all_wass_dist'] = [mean_w_dist]
        else:
            R.RESULT['all_wass_dist'].append(mean_w_dist)


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


def compute_electric_energy(sol, boxsize=50):
    def getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx):
        """
    Calculate the acceleration on each particle due to electric field
        pos      is an Nx1 matrix of particle positions
        Nx       is the number of mesh cells
        boxsize  is the domain [0,boxsize]
        n0       is the electron number density
        Gmtx     is an Nx x Nx matrix for calculating the gradient on the grid
        Lmtx     is an Nx x Nx matrix for calculating the laplacian on the grid
        a        is an Nx1 matrix of accelerations
        """
        # Calculate Electron Number Density on the Mesh by
        # placing particles into the 2 nearest bins (j & j+1, with proper weights)
        # and normalizing
        N = pos.shape[0]
        dx = boxsize / Nx
        j = np.floor(pos/dx).astype(int)
        jp1 = j+1
        weight_j = (jp1*dx - pos)/dx
        weight_jp1 = (pos - j*dx)/dx
        j = np.mod(j, Nx)   # periodic BC
        jp1 = np.mod(jp1, Nx)   # periodic BC

        n = np.bincount(j[:, 0],   weights=weight_j[:, 0],   minlength=Nx)
        n += np.bincount(jp1[:, 0], weights=weight_jp1[:, 0], minlength=Nx)
        n *= n0 * boxsize / N / dx
        n_eff = n - n0
        n_eff[-1] = 0
        n_eff = n - n0
        n_eff[-1] = 0

        # Solve Poisson's Equation: laplacian(phi) = n-n0
        phi_grid = spsolve(Lmtx, n_eff, permc_spec="MMD_AT_PLUS_A")

        # Apply Derivative to get the Electric field
        E_grid = - Gmtx @ phi_grid

        # Interpolate grid value onto particle locations
        E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]
        # interpolate grid value onto particle locations
        phi = weight_j * phi_grid[j] + weight_jp1 * phi_grid[jp1]

        a = -E

        return a, phi

    def get_gradient_matrix(Nx, boxsize):
        dx = boxsize/Nx
        e = np.ones(Nx)
        diags = np.array([-1, 1])
        vals = np.vstack((-e, e))
        Gmtx = sp.spdiags(vals, diags, Nx, Nx)
        Gmtx = sp.lil_matrix(Gmtx)
        Gmtx[0, Nx-1] = -1
        Gmtx[Nx-1, 0] = 1
        Gmtx /= (2*dx)
        Gmtx = sp.csr_matrix(Gmtx)
        return Gmtx

    def get_laplacian_matrix(Nx, boxsize):
        dx = boxsize/Nx
        e = np.ones(Nx)
        diags = np.array([-1, 0, 1])
        vals = np.vstack((e, -2*e, e))
        Lmtx = sp.spdiags(vals, diags, Nx, Nx)
        Lmtx = sp.lil_matrix(Lmtx)
        Lmtx[0, Nx-1] = 1
        Lmtx[Nx-1, 0] = 1
        Lmtx /= dx**2
        Lmtx[-1, :] = 1/Nx
        Lmtx[-1, :] = 1/Nx
        Lmtx = sp.csr_matrix(Lmtx)
        return Lmtx

    T, Nn, D = sol.shape
    N = Nn // 8

    sol = sol[:, : N * 8]

    sol = np.mod(sol, boxsize)
    # sol[sol == 0.0] = 1e-5
    # sol[sol == 1.0] = (1-1e-5)

    Lmtx = get_laplacian_matrix(N, boxsize)
    Gmtx = get_gradient_matrix(N, boxsize)

    # this will take essentially as long as the full simulation
    phi = np.zeros((T, Nn))

    for j in range(T):
        _, phi[j, :] = np.squeeze(
            getAcc(sol[j, :, 0][:, None], N, boxsize, 1, Gmtx, Lmtx)
        )

    # calculate electric energy at all times
    E = -0.5 * np.mean(phi, axis=1)

    return E


def compute_wasserstein_time(test_sol, true_sol, eps=1e-3):
    T = test_sol.shape[0]
    wdist = []
    solver = sinkhorn.Sinkhorn(max_iterations=10_000, threshold=eps)
    solver = jit(solver)

    for t in tqdm(range(T), colour='blue'):
        x = test_sol[t]
        y = true_sol[t]
        geom = pointcloud.PointCloud(x, y)
        lp = linear_problem.LinearProblem(geom)
        out = solver(lp)

        wdist_t = jnp.sqrt(out.primal_cost)
        wdist.append(wdist_t)

    # Stack into a single JAX array
    return jnp.stack(wdist, axis=0)


def average_metrics():

    # compute averages
    metrics = [
        "all_err_electric",
        "all_wass_dist"
    ]
    for metric in metrics:
        if metric in R.RESULT:
            mean_m = np.mean(R.RESULT[metric])
            R.RESULT[f"{metric}_total"] = mean_m
            print(f"{metric} final mean: {mean_m:.5f}")

    return R.RESULT[f"all_err_electric_total"]
