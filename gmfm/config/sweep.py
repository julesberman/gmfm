
# SWEEP = {
#     "dataset": "v6",
#     "net.size": "m",
#     "net.n_harmonics": "7",
#     "optimizer.pbar_delay": "100",
#     "optimizer.iters": "500_000",
#     "loss.normalize": "sym",
#     "loss.b_min": "0.1",
#     "loss.b_max": "0.5",
#     "loss.dt": 'sm_spline',
#     "loss.dt_sm": '1e-5',
#     "loss.n_functions": "25_000",
#     "loss.sigma": "1e-2",
#     "loss.reg_amt": "1e-5, 1e-4, 1e-3, 1e-2",
#     "loss.reg_type": "div, curl",
#     "loss.omega_rho": "periodic",
#     "test.t_samples": "200"
#     "test.save_trajectories": "True"
# }

SWEEP = {
    "dataset": "turb",
    "net.size": "m",
    "optimizer.pbar_delay": "200",
    "optimizer.iters": "100_000",
    "loss.normalize": "sym",
    "loss.b_min": "4.0",
    "loss.b_max": "18.0",
    "loss.dt": 'sm_spline',
    "loss.dt_sm": '1e-5',
    "loss.n_functions": "50_000",
    "loss.sigma": "0.0",
    "loss.reg_amt": "1e-5, 1e-4, 1e-3",
    "loss.reg_type": "div",
    "loss.omega_rho": "gauss",
    "sample.bs_n": "1024",
    "data.sub_x": "1",
    "latent": "True",
    "data.n_samples": "4096",
    "test.save_trajectories": "True",
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
    "mem_gb": 500,
    "gres": "gpu:1",
    "account": "extremedata",
    "partition": "gpu",
}


def get_slurm_config():

    return SLURM_CONFIG_T
