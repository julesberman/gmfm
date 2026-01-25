from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Union

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from hydra.core.config_store import ConfigStore
from jax import jacrev, jit, vmap
from tqdm import tqdm
import matplotlib.pyplot as plt
import gmfm.io.result as R
from gmfm.config.config import Config
from gmfm.config.setup import setup
from gmfm.data.dataloader import get_dataloader
from gmfm.data.get import get_dataset
from gmfm.io.save import save_results
from gmfm.loss.get import get_loss_fn

from gmfm.net.get import get_network
from gmfm.test.test import run_test
from gmfm.train.train import train_model
import numpy as np
from gmfm.test.metrics import average_metrics
from gmfm.io.load import load
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Union

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

from gmfm.config.sweep import get_slurm_config, get_sweep
from gmfm.utils.tools import epoch_time, unique_id, pshape, print_stats, normalize
from gmfm.test.metrics import compute_mean_relerr, compute_wasserstein_time_pot
from gmfm.data.turb import get_particles, enstrophy_spectral

from gmfm.config.config import toy_cfg
from einops import rearrange
from gmfm.net.mlp import DNN
from gmfm.train.adam import adam_opt

from gmfm.test.plot import plot_sde
from gmfm.config.config import Config, get_outpath
from gmfm.loss.rff import (
    grad_phi_weights,
    make_rff_params_jnp,
    rff_grad_dot_v,
    rff_laplace_phi,
    rff_phi,
    get_phi_params
)


# Uncomment for multi-run sweep
SWEEP = {}
SLURM_CONFIG = {}

# SWEEP = {
#     "dataset": "vtwo",
#     "net.size": "l",
#     "optimizer.pbar_delay": "100",
#     "optimizer.iters": "400_000",
#     "loss.normalize": "sym",
#     # "loss.b_min": "0.05, 0.1",
#     # "loss.b_max": "0.5",
#     "loss.dt": 'sm_spline',
#     "loss.dt_sm": '1e-5',
#     "loss.n_functions": "50_000",
#     "loss.sigma": "5e-2",
#     "loss.reg_amt": "0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0",
#     "loss.reg_type": "kin, traj, grad",
# }

SLURM_CONFIG_T = {
    "timeout_min": 60 * 6,
    "cpus_per_task": 4,
    "mem_gb": 50,
    "gres": "gpu:1",
    "account": "extremedata",
    "partition": "gpu",
}


defaults = [
    # https://hydra.cc/docs/tutorials/structured_config/defaults/
    # "_self_",
    {"override hydra/launcher": "submitit_slurm"},
]


hydra_config = {
    # sets the out dir from config.problem and id
    "run": {"dir": "results/${problem}/single/${name}"},
    "sweep": {"dir": "results/${problem}/multi/${name}"},
    # "mode": get_mode(),
    "sweeper": {"params": {**SWEEP}},
    # https://hydra.cc/docs/1.2/plugins/submitit_launcher/
    "launcher": {**SLURM_CONFIG},
    "job": {"env_set": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}},
    "job_logging": {"root": {"level": "WARN"}},
}


@dataclass
class ParticleExp:
    problem: str = 'particles'
    n_functions: int = 50_000
    b_min: float = 0.1
    b_max: float = 1.0
    reg_ent: float = 0.0
    reg_kin: float = 0.0
    reg_curl: float = 0.0
    reg_div: float = 0.0
    iters: int = 50_000
    n_samples: int = 25_000
    sigma: float = 0.0
    use_grad: bool = False

    # misc
    dump: bool = True
    info: Union[str, None] = None
    name: str = field(
        default_factory=lambda: f"{unique_id(4)}_{epoch_time(2)}")
    x64: bool = False  # whether to use 64 bit precision in jax
    platform: Union[str, None] = None  # gpu or cpu, None will let jax default
    seed: int = 1
    debug_nans: bool = False  # whether to debug nans
    hydra: Any = field(default_factory=lambda: hydra_config)
    defaults: List[Any] = field(default_factory=lambda: defaults)


cs = ConfigStore.instance()
cs.store(name="default_particles", node=ParticleExp)


@hydra.main(version_base=None, config_name="default_particles")
def run_particles(cfg: ParticleExp):
    key = setup(cfg)
    key, dkey, lkey = jax.random.split(key, num=3)
    N = cfg.n_samples
    T = 2.5
    viscosity = 1e-3
    max_velocity = 7
    resolution = 256
    key = jax.random.PRNGKey(0)
    Xs, Us = get_particles(0.01, N, T, viscosity,
                           max_velocity, resolution, key)
    Xs, Us = jax.device_get((Xs, Us))

    T = 256
    t_idx = np.linspace(0, Xs.shape[0] - 2, T, dtype=np.int32)
    data = Xs[t_idx]
    U_data = Us[t_idx]
    t_data = np.linspace(0, 1, T)

    true_vort, true_ent = vmap(enstrophy_spectral, (0, None))(U_data, True)

    x_data = rearrange(data, "T N ... -> N T (...)")
    n_functions = cfg.n_functions
    toy_cfg.loss.n_functions = n_functions

    toy_cfg.loss.dt = 'sm_spline'
    toy_cfg.loss.dt_sm = 1e-5
    toy_cfg.loss.b_min = cfg.b_min
    toy_cfg.loss.b_max = cfg.b_max
    mm, lhs, fixed_omega, sigma_t = get_phi_params(
        toy_cfg, x_data, t_data, dkey)

    lhs = jnp.asarray(lhs)
    fixed_omega = jnp.asarray(fixed_omega)
    data = jnp.asarray(data)
    t_data = jnp.asarray(t_data)
    true_ent = jnp.asarray(true_ent)

    @jit
    def get_batch(key):
        key1, key2, key3 = jax.random.split(key, num=3)

        t_idx = jax.random.randint(key1, shape=(), minval=0, maxval=T-1)

        x_batch0 = data[:, :]
        xt = x_batch0[t_idx]
        lhs_batch = lhs[t_idx]
        omegas = fixed_omega[:]
        lhs_batch = lhs_batch[:]

        t0 = t_data[t_idx].reshape(1, 1)
        t = jnp.repeat(t0, x_batch0.shape[1], axis=0)  # [B,1]
        dt = t_data[1]  # equispaced
        ent = true_ent[t_idx]
        return xt, t, lhs_batch, omegas, dt, ent

    cxt, ct, _, _, dt, ent = get_batch(lkey)
    if cfg.use_grad:
        net = DNN(width=128, depth=7, out_features=1,
                  residual=False, n_harmonics=1, period=jnp.pi*2)
        params_init = net.init(lkey, cxt, ct, None)

        def apply_fn(*args):
            return jnp.squeeze(vmap(jax.jacrev(net.apply, 1), (None, 0, 0, None))(*args))

    else:
        net = DNN(width=128, depth=7, out_features=2,
                  residual=False, n_harmonics=1, period=jnp.pi*2)
        params_init = net.init(lkey, cxt, ct, None)
        apply_fn = net.apply

    # Init
    compute_ent_fn = get_compute_ent(128)
    tt = jnp.mean(ct)
    tt, cxt.shape

    sigma = cfg.sigma
    loss_fn = get_loss_fn_rff(
        apply_fn,
        compute_ent_fn,
        sigma=sigma,
        lambda_ent=cfg.reg_ent,
        lambda_energy=cfg.reg_kin,
        lambda_curl=cfg.reg_curl,
        lambda_div=cfg.reg_div,
    )

    iters = cfg.iters

    opt_params, loss_history = adam_opt(
        params_init,
        loss_fn,
        get_batch,
        steps=iters,
        learning_rate=5e-4,
        key=key,
        optimizer='adam',
        loss_key=True,
        verbose=True,
        n_save=100,
        # grad_clip_norm=1.0,
    )

    key, skey = jax.random.split(key)
    x0_test = data[0, :]

    apply_fn = jit(apply_fn)
    traj = sample_trajectories(
        x0_test, apply_fn, opt_params, sigma, skey, n_steps=T)

    ents = []
    vorts = []
    for t in t_data:
        vort, ent = compute_ent_fn(apply_fn, opt_params, t, True)
        ents.append(ent)
        vorts.append(vort)
    ents = np.asarray(ents)
    vorts = np.asarray(vorts)

    pshape(traj, data)
    print(f"computing wasserstein")

    w_time = compute_wasserstein_time_pot(
        data, traj, n_samples=10_000, sub_t=32)

    R.RESULT[f"time_wass_dist"] = w_time
    mean_w_dist = np.mean(w_time)
    R.RESULT[f"mean_wass_dist"] = mean_w_dist
    print(f"mean_wass_dist: {mean_w_dist:.3e}")

    print("save...")
    R.RESULT['loss_history'] = loss_history
    R.RESULT['last_loss'] = loss_history[-1]
    R.RESULT['t_data'] = t_data
    R.RESULT['true'] = data
    R.RESULT['test'] = traj
    R.RESULT['params'] = opt_params
    R.RESULT['test_ents'] = ents
    R.RESULT['true_ents'] = true_ent

    traj = np.swapaxes(traj, 0, 1)
    data = np.swapaxes(data, 0, 1)

    traj, _ = normalize(traj, method='-11', axis=-1)
    data, _ = normalize(data, method='-11', axis=-1)
    plot_sde(cfg, traj, data)

    outdir = get_outpath()
    plt.figure(figsize=(8, 6))
    plt.plot(ents, label='Pred')
    plt.plot(true_ent, label='True')
    plt.xlabel("time")
    plt.ylabel("enstrophy")
    plt.legend()
    plt.tight_layout()
    plt.gcf().savefig(
        f"{outdir}/entr.png",
    )
    plt.cla()
    plt.clf()

    plt.figure(figsize=(8, 6))
    plt.semilogy(w_time)
    plt.xlabel("time")
    plt.ylabel("w_time")
    plt.tight_layout()
    plt.gcf().savefig(
        f"{outdir}/wass.png",
    )
    plt.cla()
    plt.clf()

    save_results(R.RESULT, cfg)

    print("done!")


def get_loss_fn_rff(
    apply_fn,
    compute_ent_fn,
    sigma: float = 0.0,
    lambda_ent: float = 0.0,
    lambda_energy: float = 0.0,
    lambda_curl: float = 0.0,
    lambda_div: float = 0.0,
):
    """
    Your weak-form loss + optional time-derivative regularization.
    """

    def loss_fn(params, xt, t, lhs, omega_, dt, true_ent, key):

        v_t = apply_fn(params, xt, t, None)  # (B, D)

        # E[∇phi · v]
        g_t = rff_grad_dot_v(xt, v_t, omega_)

        # E[Δphi]
        if sigma > 0.0:
            lap_t = rff_laplace_phi(xt, omega_)
            rhs = g_t + 0.5 * (sigma**2) * lap_t
        else:
            rhs = g_t

        err2 = (lhs - rhs)**2
        den = jnp.mean(lhs**2) + jnp.mean(rhs**2)
        den = jax.lax.stop_gradient(den)
        final_loss = jnp.mean(err2) / (den + 1e-8)

        if lambda_ent > 0:
            tt = jnp.mean(t)
            v_ent = compute_ent_fn(apply_fn, params, tt, False)
            final_loss = final_loss + lambda_ent * \
                jnp.mean((true_ent-v_ent) ** 2 / true_ent**2)

        if lambda_energy > 0:
            final_loss = final_loss + lambda_energy * \
                jnp.mean(jnp.sum(v_t ** 2, axis=-1))

        # --- Divergence regularizer (2D) ---
        if lambda_div > 0:
            def v_single(xi, ti):
                # (2,)
                return apply_fn(params, xi[None, :], jnp.expand_dims(ti, 0), None)[0]

            # Jacobian J[b, i, j] = d v_i / d x_j, shape (B, 2, 2)
            J = jax.vmap(jax.jacrev(v_single, argnums=0))(xt, t)
            div = J[:, 0, 0] + J[:, 1, 1]  # (B,)
            final_loss = final_loss + lambda_div * jnp.mean(div**2)

        # --- Curl regularizer (2D) ---
        if lambda_curl > 0:
            # scalar curl in 2D: d v_y / d x - d v_x / d y
            def v_single(xi, ti):
                # returns (2,)
                return apply_fn(params, xi[None, :], jnp.expand_dims(ti, 0), None)[0]

            # Jacobian J[i] = dv_i / d x_j, shape (B, 2, 2)
            J = jax.vmap(jax.jacrev(v_single, argnums=0))(xt, t)
            print(J.shape)
            J = jnp.squeeze(J)
            curl = J[:, 1, 0] - J[:, 0, 1]  # (B,)

            # Encourage having curl by maximizing E[curl^2]
            final_loss = final_loss - lambda_curl * jnp.mean(curl**2)

        return final_loss

    return loss_fn


def get_compute_ent(n_pts=128):
    xs = np.linspace(0, jnp.pi*2, n_pts)
    X, Y = np.meshgrid(xs, xs)
    x = np.c_[X.ravel(), Y.ravel()]
    x = jnp.asarray(x)

    def compute_ent(apply_fn, params, t, return_vort):
        t = jnp.full((x.shape[0], 1), t)
        u, v = jnp.array(apply_fn(params, x, t, None)).T
        vel = jnp.stack(
            [u.reshape(n_pts, n_pts), v.reshape(n_pts, n_pts)], axis=-1)
        return enstrophy_spectral(vel, return_vort)

    return compute_ent


def sample_trajectories(x0, apply_fn, opt_params, sigma, key, n_steps=100):
    x = x0
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
        x = (x % (jnp.pi*2))
        traj.append(x)

    return jnp.stack(traj, 0)


if __name__ == "__main__":
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    run_particles()
