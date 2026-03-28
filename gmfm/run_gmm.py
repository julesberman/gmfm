"""
NGIF scaling experiment: isotropic Gaussian mean dynamics in d dimensions.
Hydra-based single-experiment runner. Use --multirun for sweeps over (d, M, seed).
"""

import time as timer
from dataclasses import dataclass, field
from typing import Any, List, Union

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from hydra.core.config_store import ConfigStore
from jax import jit

import gmfm.io.result as R
from gmfm.config.config import get_outpath, toy_cfg
from gmfm.config.setup import setup
from gmfm.io.save import save_results
from gmfm.loss.rff import get_phi_params, rff_grad_dot_v, rff_laplace_phi
from gmfm.net.mlp import DNN
from gmfm.test.plot import get_hist
from gmfm.train.adam import adam_opt
from gmfm.utils.plot import plot_grid_movie, scatter_movie
from gmfm.utils.tools import epoch_time, unique_id

# ---------------------------------------------------------------------------
# Hydra sweep / launcher config
# ---------------------------------------------------------------------------

SWEEP = {
    "problem": "rotgauss",
    "M": '5,50,500,5000,50_000',
    "d": '2,4,8,16,32,64,128,256,512,1024',
}

SLURM_CONFIG = {
    "timeout_min": 60 * 6,
    "cpus_per_task": 4,
    "mem_gb": 200,
    "gres": "gpu:h100:1",
    "account": "extremedata",
}

defaults = [
    {"override hydra/launcher": "submitit_slurm"},
]

hydra_config = {
    "run": {"dir": "results/${problem}/single/${name}"},
    "sweep": {"dir": "results/${problem}/multi/${name}"},
    "sweeper": {"params": {**SWEEP}},
    "launcher": {**SLURM_CONFIG},
    "job": {"env_set": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}},
    "job_logging": {"root": {"level": "WARN"}},
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GMMExp:
    problem: str = 'rotgauss'
    d: int = 8
    M: int = 50_000
    seed: int = 1
    n_time_steps: int = 64       # K
    n_samples: int = 25_000      # N per timestep
    n_eval: int = 5_000
    iters: int = 250_000
    width: int = 256
    depth: int = 7
    b_min: float = 0.05
    b_max: float = 2.0
    lambda_energy: float = 0.0
    lambda_div: float = 0.0
    variance: float = 0.02
    mean_radius: float = 0.55
    odd_coord_amplitude: float = 0.4
    rotation_turns: float = 0.5
    periodic_boundary: bool = True
    periodic_half_width: Union[float, None] = None
    periodic_n_stds: float = 4.5
    periodic_margin: float = 0.15
    lr: float = 5e-4

    # misc
    dump: bool = True
    info: Union[str, None] = None
    name: str = field(
        default_factory=lambda: f"{unique_id(4)}_{epoch_time(2)}")
    x64: bool = False
    platform: Union[str, None] = None
    debug_nans: bool = False
    hydra: Any = field(default_factory=lambda: hydra_config)
    defaults: List[Any] = field(default_factory=lambda: defaults)


cs = ConfigStore.instance()
cs.store(name="default_gmm", node=GMMExp)


# ---------------------------------------------------------------------------
# Rotating Gaussian helpers
# ---------------------------------------------------------------------------

def rotation_matrix(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def rotating_gaussian_params(
    t,
    d,
    variance=0.02,
    mean_radius=0.55,
    odd_coord_amplitude=0.4,
    rotation_turns=0.5,
):
    """
    Blockwise Gaussian marginals with simple mean dynamics in every 2D pair.

    For each pair of coordinates (x_{2k}, x_{2k+1}) we rotate the block mean by
    a shared angular speed with a pair-dependent phase offset. The covariance is
    fixed over time. If d is odd, the last coordinate undergoes a 1D sinusoidal
    shift with fixed variance.
    """
    t = float(np.clip(t, 0.0, 1.0))
    angle = 2.0 * np.pi * rotation_turns * t
    n_pairs = d // 2

    mean = np.zeros(d, dtype=np.float64)
    cov = np.zeros((d, d), dtype=np.float64)
    base_mean = np.asarray([mean_radius, 0.35 * mean_radius], dtype=np.float64)
    base_cov = variance * np.eye(2, dtype=np.float64)

    for pair_idx in range(n_pairs):
        phase = pair_idx * (np.pi / 7.0)
        rot = rotation_matrix(angle + phase)
        pair_mean = rot @ base_mean
        sl = slice(2 * pair_idx, 2 * pair_idx + 2)
        mean[sl] = pair_mean
        cov[sl, sl] = base_cov

    if d % 2 == 1:
        mean[-1] = odd_coord_amplitude * np.sin(angle)
        cov[-1, -1] = variance

    return mean, cov


def sample_rotating_gaussian(
    n_samples,
    t,
    d=2,
    variance=0.02,
    mean_radius=0.55,
    odd_coord_amplitude=0.4,
    rotation_turns=0.5,
    rng=None,
):
    """
    Sample from the rotating Gaussian marginals at time t.
    """
    if rng is None or isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)

    t = float(np.clip(t, 0.0, 1.0))
    angle = 2.0 * np.pi * rotation_turns * t
    n_pairs = d // 2
    samples = np.zeros((n_samples, d), dtype=np.float32)
    base_mean = np.asarray([mean_radius, 0.35 * mean_radius], dtype=np.float64)
    base_cov = variance * np.eye(2, dtype=np.float64)

    for pair_idx in range(n_pairs):
        phase = pair_idx * (np.pi / 7.0)
        rot = rotation_matrix(angle + phase)
        pair_mean = rot @ base_mean
        samples[:, 2 * pair_idx:2 * pair_idx + 2] = rng.multivariate_normal(
            mean=pair_mean,
            cov=base_cov,
            size=n_samples,
        ).astype(np.float32)

    if d % 2 == 1:
        samples[:, -1] = (
            odd_coord_amplitude * np.sin(angle)
            + rng.normal(0.0, np.sqrt(variance), size=n_samples)
        ).astype(np.float32)

    return samples


def generate_rotgauss_data(
    d,
    K,
    N,
    variance,
    mean_radius,
    odd_coord_amplitude,
    rotation_turns,
    seed=0,
):
    """
    Generate samples from the rotating Gaussian process at K time steps.
    """
    rng = np.random.default_rng(seed)
    t_data = np.linspace(0, 1, K)
    data = np.empty((K, N, d), dtype=np.float32)
    for k in range(K):
        data[k] = sample_rotating_gaussian(
            N,
            t_data[k],
            d=d,
            variance=variance,
            mean_radius=mean_radius,
            odd_coord_amplitude=odd_coord_amplitude,
            rotation_turns=rotation_turns,
            rng=rng,
        )
    return data, t_data


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_nll(
    samples,
    t,
    d,
    variance,
    mean_radius,
    odd_coord_amplitude,
    rotation_turns,
):
    """
    Closed-form mean negative log-likelihood under the true rotating Gaussian.
    """
    t = jnp.clip(jnp.asarray(t), 0.0, 1.0)
    angle = 2.0 * jnp.pi * rotation_turns * t
    n_pairs = d // 2
    nll = jnp.zeros(samples.shape[0], dtype=samples.dtype)

    if n_pairs > 0:
        x_pairs = samples[:, :2 *
                          n_pairs].reshape(samples.shape[0], n_pairs, 2)
        pair_idx = jnp.arange(n_pairs, dtype=samples.dtype)
        phase = pair_idx * (jnp.pi / 7.0)
        total_angle = angle + phase
        c = jnp.cos(total_angle)
        s = jnp.sin(total_angle)

        base_mean = jnp.asarray(
            [mean_radius, 0.35 * mean_radius], dtype=samples.dtype)
        mean_pairs = jnp.stack(
            [
                c * base_mean[0] - s * base_mean[1],
                s * base_mean[0] + c * base_mean[1],
            ],
            axis=-1,
        )

        centered = x_pairs - mean_pairs[None, :, :]
        quad = jnp.sum(centered ** 2, axis=-1) / variance
        pair_log_norm = jnp.log(2 * jnp.pi * variance)
        nll = nll + jnp.sum(pair_log_norm + 0.5 * quad, axis=-1)

    if d % 2 == 1:
        mean_last = odd_coord_amplitude * jnp.sin(angle)
        centered_last = samples[:, -1] - mean_last
        nll = nll + 0.5 * (
            jnp.log(2 * jnp.pi * variance) + (centered_last ** 2) / variance
        )

    return jnp.mean(nll)


def compute_true_mean(
    t,
    d,
    mean_radius,
    odd_coord_amplitude,
    rotation_turns,
    dtype=jnp.float32,
):
    """
    Analytic mean of the rotating Gaussian at time t.
    """
    t = jnp.clip(jnp.asarray(t, dtype=dtype), 0.0, 1.0)
    angle = 2.0 * jnp.pi * rotation_turns * t
    n_pairs = d // 2
    mean = jnp.zeros((d,), dtype=dtype)

    if n_pairs > 0:
        pair_idx = jnp.arange(n_pairs, dtype=dtype)
        phase = pair_idx * (jnp.pi / 7.0)
        total_angle = angle + phase
        c = jnp.cos(total_angle)
        s = jnp.sin(total_angle)
        base_mean = jnp.asarray(
            [mean_radius, 0.35 * mean_radius], dtype=dtype)
        mean_pairs = jnp.stack(
            [
                c * base_mean[0] - s * base_mean[1],
                s * base_mean[0] + c * base_mean[1],
            ],
            axis=-1,
        ).reshape(-1)
        mean = mean.at[:2 * n_pairs].set(mean_pairs)

    if d % 2 == 1:
        mean = mean.at[-1].set(odd_coord_amplitude * jnp.sin(angle))

    return mean


def compute_mean_relerr(
    samples,
    t,
    d,
    mean_radius,
    odd_coord_amplitude,
    rotation_turns,
    eps=1e-8,
):
    """
    Mean absolute relative error of the per-dimension sample mean.
    """
    pred_mean = jnp.mean(samples, axis=0)
    true_mean = compute_true_mean(
        t,
        d,
        mean_radius,
        odd_coord_amplitude,
        rotation_turns,
        dtype=samples.dtype,
    )
    denom = jnp.maximum(jnp.abs(true_mean),
                        jnp.asarray(eps, dtype=samples.dtype))
    relerr = jnp.abs(pred_mean - true_mean) / denom
    return jnp.mean(relerr)


def sliced_wasserstein(x, y, n_proj=50, key=None):
    """
    Sliced Wasserstein distance between two point clouds.
    x, y: (N, d)
    """
    d = x.shape[-1]
    dirs = jax.random.normal(key, shape=(n_proj, d))
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)  # (n_proj, d)
    px = x @ dirs.T  # (N, n_proj)
    py = y @ dirs.T  # (N, n_proj)
    px_sorted = jnp.sort(px, axis=0)
    py_sorted = jnp.sort(py, axis=0)
    return jnp.mean(jnp.abs(px_sorted - py_sorted))


# ---------------------------------------------------------------------------
# Trajectory integration with optional periodic wrapping
# ---------------------------------------------------------------------------

def get_periodic_half_width(
    variance,
    mean_radius,
    odd_coord_amplitude,
    n_stds=4.5,
    margin=0.15,
):
    """
    Choose a box that is tight for the rotating-Gaussian support but still wide
    enough that true samples rarely touch the boundary.
    """
    base_mean = np.asarray([mean_radius, 0.35 * mean_radius], dtype=np.float64)
    max_pair_coord = np.linalg.norm(base_mean)
    max_coord = max(max_pair_coord, abs(odd_coord_amplitude))
    return float(max_coord + n_stds * np.sqrt(variance) + margin)


def wrap_periodic(x, half_width):
    if half_width is None:
        return x

    half_width = jnp.asarray(half_width, dtype=x.dtype)
    box_width = 2.0 * half_width
    return jnp.mod(x + half_width, box_width) - half_width


def sample_trajectories(
    x0,
    apply_fn,
    opt_params,
    sigma,
    key,
    n_steps=100,
    periodic_half_width=None,
):
    x = wrap_periodic(x0, periodic_half_width)
    ts = jnp.linspace(0.0, 1.0, n_steps)
    dt = ts[1] - ts[0]
    traj = [x]
    batch_size = x0.shape[0]
    for ti in ts[:-1]:
        skey, key = jax.random.split(key)
        eps = jax.random.normal(skey, shape=x.shape)
        t_batch = jnp.full((batch_size, 1), ti)
        v = apply_fn(opt_params, x, t_batch, None)
        x = x + v * dt + sigma * jnp.sqrt(dt) * eps
        x = wrap_periodic(x, periodic_half_width)
        traj.append(x)
    return jnp.stack(traj, 0)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def get_loss_fn_rff_rotgauss(
    apply_fn,
    sigma: float = 0.0,
    lambda_energy: float = 0.0,
    lambda_div: float = 0.0,
):
    def loss_fn(params, xt, t, lhs, omega_, dt, _unused_ent, key):
        v_t = apply_fn(params, xt, t, None)  # (B, D)

        g_t = rff_grad_dot_v(xt, v_t, omega_)

        if sigma > 0.0:
            lap_t = rff_laplace_phi(xt, omega_)
            rhs = g_t + 0.5 * (sigma ** 2) * lap_t
        else:
            rhs = g_t

        err2 = (lhs - rhs) ** 2
        den = jnp.mean(lhs ** 2) + jnp.mean(rhs ** 2)
        den = jax.lax.stop_gradient(den)
        final_loss = jnp.mean(err2) / (den + 1e-8)

        if lambda_energy > 0:
            final_loss = final_loss + lambda_energy * \
                jnp.mean(jnp.sum(v_t ** 2, axis=-1))

        if lambda_div > 0:
            def v_single(xi, ti):
                return apply_fn(params, xi[None, :], jnp.expand_dims(ti, 0), None)[0]
            J = jax.vmap(jax.jacrev(v_single, argnums=0))(xt, t)
            div = jnp.trace(J, axis1=-2, axis2=-1)  # (B,)
            final_loss = final_loss + lambda_div * jnp.mean(div ** 2)

        return final_loss

    return loss_fn


def plot_pair_scatter_movies(true_data, pred_data, outpath, frames, max_pairs=5):
    n_pairs = min(true_data.shape[-1] // 2, max_pairs)
    if n_pairs == 0:
        return

    for pair_idx in range(n_pairs):
        d0 = 2 * pair_idx
        d1 = d0 + 1
        pair_true = true_data[:, :, [d0, d1]]
        pair_pred = pred_data[:, :, [d0, d1]]
        pair_sol = np.asarray([pair_true, pair_pred])
        scatter_movie(
            pair_sol,
            alpha=0.3,
            show=False,
            frames=frames,
            title=f"dims {d0}-{d1}",
            save_to=outpath / f"pair_{d0:02d}_{d1:02d}",
        )


# ---------------------------------------------------------------------------
# Main Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_name="default_gmm")
def run_gmm(cfg: GMMExp):
    key = setup(cfg)
    key, lkey = jax.random.split(key)

    d = cfg.d
    M = cfg.M
    K = cfg.n_time_steps
    N = cfg.n_samples
    variance = cfg.variance
    mean_radius = cfg.mean_radius
    odd_coord_amplitude = cfg.odd_coord_amplitude
    rotation_turns = cfg.rotation_turns
    periodic_half_width = None
    if cfg.periodic_boundary:
        periodic_half_width = cfg.periodic_half_width
        if periodic_half_width is None:
            periodic_half_width = get_periodic_half_width(
                variance=variance,
                mean_radius=mean_radius,
                odd_coord_amplitude=odd_coord_amplitude,
                n_stds=cfg.periodic_n_stds,
                margin=cfg.periodic_margin,
            )

    print(f"\n{'='*60}")
    print(f"d={d}, M={M}, seed={cfg.seed}")
    print(f"{'='*60}")
    if periodic_half_width is not None:
        print(
            f"Periodic rollout box: [-{periodic_half_width:.3f}, {periodic_half_width:.3f}]^d")

    # --- Generate data ---
    data, t_data = generate_rotgauss_data(
        d,
        K,
        N,
        variance,
        mean_radius,
        odd_coord_amplitude,
        rotation_turns,
        seed=cfg.seed,
    )
    x_data = rearrange(data, "T N D -> N T D")

    # --- RFF setup ---
    toy_cfg.loss.n_functions = M
    toy_cfg.loss.dt = 'sm_spline'
    toy_cfg.loss.dt_sm = 1e-5
    toy_cfg.loss.b_min = cfg.b_min
    toy_cfg.loss.b_max = cfg.b_max

    key, phi_key = jax.random.split(key)
    mm, lhs, fixed_omega, sigma_t = get_phi_params(
        toy_cfg, x_data, t_data, phi_key)

    lhs = jnp.asarray(lhs)
    fixed_omega = jnp.asarray(fixed_omega)
    data_jnp = jnp.asarray(data)
    t_data_jnp = jnp.asarray(t_data)

    T = K

    @jit
    def get_batch(key):
        key1, key2 = jax.random.split(key)
        t_idx = jax.random.randint(key1, shape=(), minval=0, maxval=T - 1)
        xt = data_jnp[t_idx]
        lhs_batch = lhs[t_idx]
        omegas = fixed_omega[:]
        t0 = t_data_jnp[t_idx].reshape(1, 1)
        t = jnp.repeat(t0, xt.shape[0], axis=0)  # (N, 1)
        dt = t_data_jnp[1]
        ent = jnp.array(0.0)  # placeholder
        return xt, t, lhs_batch, omegas, dt, ent

    # --- Network ---
    cxt, ct, _, _, _, _ = get_batch(lkey)
    net = DNN(width=cfg.width, depth=cfg.depth, out_features=d,
              residual=False, n_harmonics=0, period=1.0)
    params_init = net.init(lkey, cxt, ct, None)
    apply_fn = net.apply

    # --- Loss ---
    loss_fn = get_loss_fn_rff_rotgauss(
        apply_fn,
        sigma=0.0,
        lambda_energy=cfg.lambda_energy,
        lambda_div=cfg.lambda_div,
    )

    # --- Train ---
    t_start = timer.time()
    opt_params, loss_history = adam_opt(
        params_init,
        loss_fn,
        get_batch,
        steps=cfg.iters,
        learning_rate=cfg.lr,
        key=key,
        optimizer='adam',
        loss_key=True,
        verbose=True,
        n_save=100,
    )
    train_time = timer.time() - t_start
    print(f"Training time: {train_time:.1f}s")

    # --- Evaluate ---
    apply_fn_jit = jit(apply_fn)

    key, traj_key = jax.random.split(key)
    N_eval = cfg.n_eval
    eval_rng = np.random.default_rng(cfg.seed + 1)
    x0_eval = jnp.asarray(
        sample_rotating_gaussian(
            N_eval,
            0.0,
            d=d,
            variance=variance,
            mean_radius=mean_radius,
            odd_coord_amplitude=odd_coord_amplitude,
            rotation_turns=rotation_turns,
            rng=eval_rng,
        )
    )

    n_eval_steps = K
    traj = sample_trajectories(
        x0_eval,
        apply_fn_jit,
        opt_params,
        0.0,
        traj_key,
        n_steps=n_eval_steps,
        periodic_half_width=periodic_half_width,
    )

    eval_ts = jnp.linspace(0.0, 1.0, n_eval_steps)
    nlls = []
    swds = []
    mean_relerrs = []
    for k_idx in range(1, n_eval_steps):
        tk = float(eval_ts[k_idx])
        samples_k = traj[k_idx]  # (N_eval, d)

        nll_k = compute_nll(
            samples_k,
            tk,
            d,
            variance,
            mean_radius,
            odd_coord_amplitude,
            rotation_turns,
        )
        nlls.append(float(nll_k))

        mean_relerr_k = compute_mean_relerr(
            samples_k,
            tk,
            d,
            mean_radius,
            odd_coord_amplitude,
            rotation_turns,
        )
        mean_relerrs.append(float(mean_relerr_k))

        key, swd_key = jax.random.split(key)
        true_samples_k = jnp.asarray(
            sample_rotating_gaussian(
                N_eval,
                tk,
                d=d,
                variance=variance,
                mean_radius=mean_radius,
                odd_coord_amplitude=odd_coord_amplitude,
                rotation_turns=rotation_turns,
                rng=eval_rng,
            )
        )
        swd_k = sliced_wasserstein(
            samples_k, true_samples_k, n_proj=50, key=swd_key)
        swds.append(float(swd_k))

    mean_nll = np.mean(nlls)
    mean_swd = np.mean(swds)
    mean_mean_relerr = np.mean(mean_relerrs)
    print(
        "Mean NLL: "
        f"{mean_nll:.4f}, "
        f"Mean SWD: {mean_swd:.6f}, "
        f"Mean Mean RelErr: {mean_mean_relerr:.6f}, "
        f"Time: {train_time:.1f}s"
    )

    # --- GIF animation (first 2 spatial dims) ---
    outpath = get_outpath()
    outpath.mkdir(exist_ok=True, parents=True)
    frames = 75

    true_np = np.asarray(data)       # (K, N, d)
    pred_np = np.asarray(traj)       # (K, N_eval, d)

    if periodic_half_width is not None:
        true_np = np.asarray(wrap_periodic(
            jnp.asarray(true_np), periodic_half_width))
        pred_np = np.asarray(wrap_periodic(
            jnp.asarray(pred_np), periodic_half_width))

    # Subsample true to match N_eval so arrays can be stacked
    n_plot = min(true_np.shape[1], pred_np.shape[1])
    true_np = true_np[:, :n_plot, :]
    pred_np = pred_np[:, :n_plot, :]

    if d > 2:
        true_2d = true_np[:, :, :2]
        pred_2d = pred_np[:, :, :2]
    else:
        true_2d = true_np
        pred_2d = pred_np

    try:
        plot_sol = np.asarray([true_2d, pred_2d])  # (2, K, n_plot, 2)
        scatter_movie(plot_sol, alpha=0.3,
                      show=False, frames=frames,
                      save_to=f'{outpath}/sol.gif')
    except Exception as e:
        print(e, "could not plot scatter gif")

    try:
        idx_time = np.linspace(0, len(pred_2d) - 1, frames, dtype=np.int32)
        hist_true = get_hist(true_2d[idx_time])
        hist_pred = get_hist(pred_2d[idx_time])
        plot_grid_movie([hist_true, hist_pred], frames=frames, show=False,
                        save_to=f'{outpath}/hist.gif',
                        titles_x=['True', 'Pred'], live_cbar=True)
    except Exception as e:
        print(e, "could not plot hist gif")

    try:
        plot_pair_scatter_movies(
            true_np, pred_np, outpath, frames=frames, max_pairs=5)
    except Exception as e:
        print(e, "could not plot pair scatter gifs")

    # --- Save results ---
    R.RESULT['d'] = d
    R.RESULT['M'] = M
    R.RESULT['mean_nll'] = mean_nll
    R.RESULT['mean_swd'] = mean_swd
    R.RESULT['mean_mean_relerr'] = mean_mean_relerr
    R.RESULT['train_time_sec'] = train_time
    R.RESULT['periodic_half_width'] = periodic_half_width
    R.RESULT['loss_history'] = loss_history
    R.RESULT['last_loss'] = loss_history[-1]
    R.RESULT['nlls'] = nlls
    R.RESULT['swds'] = swds
    R.RESULT['mean_relerrs'] = mean_relerrs
    R.RESULT['params'] = opt_params

    save_results(R.RESULT, cfg)
    print("done!")


if __name__ == "__main__":
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    run_gmm()
