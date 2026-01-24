
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

SWEEP = {
    "dataset": "v6",
    "net.size": "l",
    "net.n_harmonics": "0, 1, 2, 4, 8",
    "optimizer.pbar_delay": "100",
    "optimizer.iters": "250_000",
    "loss.normalize": "sym",
    "loss.b_min": "0.05, 0.1",
    "loss.b_max": "0.5, 1.0, 2.0",
    "loss.dt": 'sm_spline',
    "loss.dt_sm": '5e-5',
    "loss.n_functions": "50_000",
    "loss.sigma": "5e-2",
    "loss.reg_amt": "1e-3",
    "loss.reg_type": "kin",
    "loss.omega_rho": "periodic"
}


# SWEEP = {
#     "dataset": "wave",
#     "net.size": "m",
#     "optimizer.pbar_delay": "100",
#     "optimizer.iters": "100_000",
#     "loss.normalize": "sym",
#     "loss.b_min": "5.0",
#     "loss.b_max": "5.0, 10.0, 20.0",
#     "loss.dt": 'sm_spline',
#     "loss.dt_sm": '1e-4',
#     "loss.n_functions": "50_000",
#     "loss.sigma": "0.0",
#     "loss.reg_amt": "0.0",
#     "loss.reg_type": "kin",
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
    "timeout_min": 60 * 8,
    "cpus_per_task": 4,
    "mem_gb": 50,
    "gres": "gpu:1",
    "account": "extremedata",
    "partition": "gpu",

}


def get_slurm_config():

    return SLURM_CONFIG_T
