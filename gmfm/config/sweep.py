
SWEEP = {
    "dataset": "v6",
    "net.size": "l",
    "optimizer.pbar_delay": "100",
    "optimizer.iters": "250_000",
    # "loss.relative": "True, False",
    "loss.normalize": "sym",
    # "loss.resample":z "True, False",
    "loss.b_min": "0.1",
    "loss.b_max": "0.5",
    "loss.dt": 'sm_spline',
    "loss.dt_sm": '1e-5',
    "loss.n_functions": "50_000, 100_000",
    "loss.sigma": "5e-2",
    "loss.reg_kin": "1e-3",
    "data.sub_t": "1"
}


# SWEEP = {
#     "dataset": "vtwo",
#     "net.size": "l",
#     "optimizer.pbar_delay": "100",
#     "optimizer.iters": "500_000",
#     "loss.relative": "True",
#     "loss.normalize": "none, omega",
#     "loss.resample": "True, False",
#     "loss.b_min": "0.005, 0.01, 0.1",
#     "loss.b_max": "0.5, 1.0",
#     "loss.dt": 'cubic',
#     "loss.omega_rho": 'orf',
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
    "timeout_min": 60 * 16,
    "cpus_per_task": 4,
    "mem_gb": 50,
    "gres": "gpu:1",
    "account": "extremedata",
    "partition": "gpu",

}


def get_slurm_config():

    return SLURM_CONFIG_T
