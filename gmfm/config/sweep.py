
SWEEP = {
    "dataset": "v6",
    "net.size": "l",
    "optimizer.pbar_delay": "100",
    "optimizer.iters": "1_000_000",
    "loss.relative": "True",
    "loss.normalize": "omega, grad",
    # "loss.resample":z "True, False",
    # "loss.bandwidths": "[0.8],[0.7],[0.6],[0.5],[0.4],[0.3],[0.2],[0.1]",

    "loss.b_min": "0.01",
    "loss.b_max": "1.0",
    "loss.dt": 'cubic',
    # "loss.omega_rho": 'orf',
    "loss.n_functions": "100_000, 200_000",
    "loss.sigma": "5e-2",
    "loss.reg_kin": "0.0, 1e-2",
}


# SWEEP = {
#     "dataset": "vtwo",
#     "net.size": "l",
#     "optimizer.pbar_delay": "100",
#     "optimizer.iters": "500_000",
#     "loss.relative": "True",
#     "loss.normalize": "none, omega",
#     # "loss.resample": "True, False",
#     "loss.b_min": "0.005, 0.01, 0.1",
#     "loss.b_max": "0.5, 1.0",
#     "loss.dt": 'cubic',
#     # "loss.omega_rho": 'orf',
#     "loss.n_functions": "100_000",
#     "loss.sigma": "5e-2",
#     "loss.reg_kin": "1e-2"
# }


def get_sweep():
    return SWEEP


# SLURM_CONFIG_K = {
#     "timeout_min": 60 * 3,
#     "cpus_per_task": 16,
#     "mem_gb": 500,
#     "gres": "gpu:h100:1",
#     "account": "torch_pr_34_bpeher",
# }


SLURM_CONFIG_T = {
    "timeout_min": 60 * 24,
    "cpus_per_task": 16,
    "mem_gb": 500,
    "gres": "gpu:1",
    "account": "torch_pr_34_bpeher",
}


def get_slurm_config():

    return SLURM_CONFIG_T
