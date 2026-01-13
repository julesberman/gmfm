import numpy as np

from gmfm.config.config import Config


def get_dataloader(
    cfg: Config,
    x_data,
    phi_data,
    lhs_data,
    t_data,
    sigmas
):
    x_data = np.asarray(x_data)
    x_data = np.ascontiguousarray(x_data)
    N, T = x_data.shape[:2]

    bs_n = cfg.sample.bs_n
    bs_o = cfg.sample.bs_o
    n_functions = cfg.loss.n_functions
    steps = cfg.optimizer.iters
    resample = cfg.loss.resample
    rng = np.random.default_rng(cfg.seed)

    if resample:
        def iterator():
            for _ in range(steps+100):
                t_idx = rng.integers(1, T)
                idx_n = rng.choice(N, size=bs_n, replace=False)

                xt_batch = x_data[idx_n, t_idx, :]
                xt_m1_batch = x_data[idx_n, t_idx-1, :]

                t = np.asarray(t_data[t_idx]).reshape(1, 1)
                t = np.repeat(t, bs_n, axis=0)

                tm1 = np.asarray(t_data[t_idx-1]).reshape(1, 1)
                tm1 = np.repeat(tm1, bs_n, axis=0)

                dt = t - tm1

                sigma_t = sigmas[t_idx].reshape(1, 1)
                sigma_t = np.repeat(sigma_t, bs_n, axis=0)

                yield xt_batch, t, xt_m1_batch, sigma_t, dt
    else:
        def iterator():
            for _ in range(steps+100):
                t_idx = rng.integers(0, T)
                idx_n = rng.choice(N, size=bs_n, replace=False)

                if bs_o > 0:
                    idx_o = rng.choice(n_functions, size=bs_o, replace=False)
                    phi_batch = phi_data[idx_o]
                    idx_o = np.concatenate([idx_o, idx_o+n_functions])
                    lhs_batch = lhs_data[t_idx, idx_o]
                elif bs_o == -1:
                    phi_batch = phi_data[:]
                    lhs_batch = lhs_data[t_idx, :]

                xt_batch = x_data[idx_n, t_idx, :]
                t0 = np.asarray(t_data[t_idx]).reshape(1, 1)
                t = np.repeat(t0, bs_n, axis=0)

                yield xt_batch, t, phi_batch, lhs_batch

    return iterator()
