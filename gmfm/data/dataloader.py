import numpy as np

from gmfm.config.config import Config


def get_dataloader(
    cfg: Config,
    x_data,
    phi_data,
    lhs_data,
    t_data,
):
    x_data = np.asarray(x_data)
    x_data = np.ascontiguousarray(x_data)
    N, T = x_data.shape[:2]

    bs_n = cfg.sample.bs_n
    bs_o = cfg.sample.bs_o
    n_functions = cfg.loss.n_functions
    stride = cfg.loss.stride
    steps = cfg.optimizer.iters
    replace = cfg.sample.replace
    rng = np.random.default_rng(cfg.seed)
    dt = t_data[stride] - t_data[0]

    def iterator():
        for _ in range(steps+100):
            t_idx = rng.integers(0, T)

            if replace:
                idx_n = rng.integers(0, N, size=bs_n)
            else:
                idx_n = rng.choice(N, size=bs_n, replace=False)

            if bs_o > 0:
                if replace:
                    idx_o = rng.integers(0, n_functions, size=bs_o)
                else:
                    idx_o = rng.choice(n_functions, size=bs_o, replace=False)
                phi_batch = phi_data[idx_o]
                if cfg.loss.basis == 'rff':
                    idx_o = np.concatenate([idx_o, idx_o+n_functions])
                lhs_batch = lhs_data[t_idx, idx_o]
            elif bs_o == -1:
                phi_batch = phi_data[:]
                lhs_batch = lhs_data[t_idx, :]

            xt_batch = x_data[idx_n, t_idx, :]
            t0 = np.asarray(t_data[t_idx]).reshape(1, 1)
            t = np.repeat(t0, bs_n, axis=0)

            yield xt_batch, phi_batch, lhs_batch, t, dt

    return iterator()
