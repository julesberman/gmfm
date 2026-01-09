SWEEP = {
    "dataset": "adv",
    "net.size": "m",
    "optimizer.pbar_delay": "200",
    "optimizer.iters": "100_000",
    "loss.relative": "True",

    "loss.bandwidths": "[16.0], [32.0, 16.0, 8.0], [8.0, 4.0, 2.0]",
    "loss.basis": "rff",
    "loss.n_functions": "5_000, 200_000",
    "data.sub_x": "1"

    # "loss.sigma": "0.0, 5e-2",
    # "loss.reg_kin": "0.0, 1e-2"
}


def get_sweep():
    return SWEEP


SLURM_CONFIG_M = {
    "timeout_min": 60 * 16,
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
