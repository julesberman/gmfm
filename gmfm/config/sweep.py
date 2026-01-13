SWEEP = {
    "dataset": "turb",
    # "net.size": "l",
    "optimizer.pbar_delay": "100",
    "optimizer.iters": "250_000",
    "loss.relative": "True",
    "loss.normalize": "none",
    # "loss.resample": "True, False",
    "loss.b_min": "1.0, 6.0",
    "loss.b_max": "10.0, 25.0",
    "loss.dt": 'cubic, akima, sm_spline, sm_fd'
    # "loss.omega_rho": 'orf',
    # "loss.n_functions": "200_000",
    # "sample.bs_n": "512"
    # "loss.sigma": "0.0, 5e-2",
    # "loss.reg_kin": "0.0, 1e-2"
}


def get_sweep():
    return SWEEP


# SLURM_CONFIG_K = {
#     "timeout_min": 60 * 3,
#     "cpus_per_task": 16,
#     "mem_gb": 500,
#     "gres": "gpu:h100:1",
#     "account": "torch_pr_34_bpeher ",
# }


SLURM_CONFIG_T = {
    "timeout_min": 60 * 8,
    "cpus_per_task": 16,
    "mem_gb": 500,
    "gres": "gpu:h200:4",
    "account": "torch_pr_34_bpeher",
}


def get_slurm_config():

    return SLURM_CONFIG_T
