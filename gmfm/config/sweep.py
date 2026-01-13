SWEEP = {
    "dataset": "turb",
    # "net.size": "l",
    "optimizer.pbar_delay": "100",
    "optimizer.iters": "200_000",
    "loss.relative": "True",
    "loss.normalize": "none, omega, grad",
    # "loss.resample": "True, False",
    "loss.bandwidths": "[21.0, 14.0, 7.0], [21.0, 14.0, 7.0, 2.0], [21.0, 14.0, 7.0, 1.0, 0.5, 0.1], [21.0, 14.0, 7.0, 1.0, 0.5, 0.1, 0.05, 0.01]",
    "loss.loss.omega_rho": 'orf',

    # "loss.n_functions": "200_000",
    # "sample.bs_n": "512"
    # "loss.sigma": "0.0, 5e-2",
    # "loss.reg_kin": "0.0, 1e-2"
}


def get_sweep():
    return SWEEP


SLURM_CONFIG_M = {
    "timeout_min": 60 * 3,
    "cpus_per_task": 16,
    "mem_gb": 500,
    "gres": "gpu:h100:1",
    "account": "extremedata",
}


SLURM_CONFIG_L = {
    "timeout_min": 60 * 16,
    "cpus_per_task": 16,
    "mem_gb": 300,
    "gres": "gpu:h100:4",
    "account": "extremedata",
}


def get_slurm_config():

    return SLURM_CONFIG_L
