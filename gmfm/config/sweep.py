
SWEEP = {
    "dataset": "vtwo",
    "net.size": "l",
    "optimizer.pbar_delay": "100",
    "optimizer.iters": "500_000",
    "loss.normalize": "sym",
    # "loss.b_min": "0.05, 0.1",
    # "loss.b_max": "0.5",
    "loss.dt": 'sm_spline',
    "loss.dt_sm": '1e-5',
    "loss.n_functions": "50_000",
    "loss.sigma": "5e-2",
    "loss.reg_amt": "0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0",
    "loss.reg_type": "kin, traj, grad",
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
