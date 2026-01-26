
# SWEEP = {
#     "dataset": "vbump",
#     "net.size": "m, l",
#     "net.n_harmonics": "4, 7",
#     "optimizer.pbar_delay": "100",
#     "optimizer.iters": "750_000",
#     "loss.normalize": "sym",
#     "loss.b_min": "0.05",
#     "loss.b_max": "1.0",
#     "loss.dt": 'sm_spline',
#     "loss.dt_sm": '1e-5',
#     "loss.n_functions": "100_000",
#     "loss.sigma": "5e-2",
#     "loss.reg_amt": "1e-2",
#     "loss.reg_type": "kin",
#     "loss.omega_rho": "periodic",
#     "test.save_trajectories": "True"
# }

SWEEP = {
    "dataset": "turb",
    "net.size": "m",
    "optimizer.pbar_delay": "200",
    "optimizer.iters": "200_000",
    "loss.normalize": "sym",
    "loss.b_min": "0.5, 1.0, 4.0",
    "loss.b_max": "1.0, 8.0, 16.0",
    "loss.dt": 'sm_spline',
    "loss.dt_sm": '1e-7, 1e-5',
    "loss.n_functions": "50_000",
    "loss.sigma": "0.0",
    "loss.reg_amt": "0.0",
    "loss.reg_type": "kin",
    "loss.omega_rho": "orf",
    "sample.bs_n": "-1",
    "data.sub_x": "1, 2",
    "latent": "True"
}


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
    "timeout_min": 60 * 18,
    "cpus_per_task": 4,
    "mem_gb": 50,
    "gres": "gpu:1",
    "account": "extremedata",
    "partition": "gpu",

}


def get_slurm_config():

    return SLURM_CONFIG_T
