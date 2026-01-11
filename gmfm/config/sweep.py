SWEEP = {
    "dataset": "turb",
    "net.size": "m",
    "optimizer.pbar_delay": "200",
    "optimizer.iters": "200_000",
    "loss.relative": "True",

    "loss.bandwidths": "[32.0, 16.0, 8.0], [32.0, 16.0, 4.0, 1.0, 0.1], [20.0, 16.0, 12.0], [24.0], [16.0], [8.0], [4.0], [1.0]",
    "loss.basis": "rff",
    "loss.n_functions": "250_000",
    "data.sub_x": "1"

    # "loss.sigma": "0.0, 5e-2",
    # "loss.reg_kin": "0.0, 1e-2"
}


def get_sweep():
    return SWEEP


SLURM_CONFIG_M = {
    "timeout_min": 60 * 26,
    "cpus_per_task": 16,
    "mem_gb": 500,
    "gres": "gpu:h100:1",
    "account": "extremedata",
}


SLURM_CONFIG_L = {
    "timeout_min": 60 * 8,
    "cpus_per_task": 16,
    "mem_gb": 500,
    "gres": "gpu:h100:4",
    "account": "extremedata",
}


def get_slurm_config():

    return SLURM_CONFIG_M
