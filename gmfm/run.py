import os

import hydra
import jax

import gmfm.io.result as R
from gmfm.config.config import Config
from gmfm.config.setup import setup
from gmfm.data.dataloader import get_dataloader
from gmfm.data.get import get_dataset
from gmfm.io.save import save_results
from gmfm.loss.get import get_loss_fn
from gmfm.loss.rff import get_phi_params
from gmfm.net.get import get_network
from gmfm.test.test import run_test
from gmfm.train.train import train_model
import numpy as np
from gmfm.test.metrics import average_metrics
from gmfm.io.load import load


@hydra.main(version_base=None, config_name="default")
def run(cfg: Config) -> None:

    if cfg.retest is not None:
        cfg, df = load(cfg.retest)
        opt_params = df['opt_params']
        key, x_data, dataloader, moments, lhs_data, phi_data, mu_data, sigma_t, loss_fn, net, params_init, apply_fn = build(
            cfg)
        cfg.loss.n_functions = 10
        key, key_G, key_test, key_opt = jax.random.split(key, num=4)
    else:
        key, x_data, dataloader, moments, lhs_data, phi_data, mu_data, sigma_t, loss_fn, net, params_init, apply_fn = build(
            cfg)

        key, key_G, key_test, key_opt = jax.random.split(key, num=4)

        opt_params = train_model(cfg, dataloader,
                                 loss_fn, params_init, key_opt, has_aux=True)

    if cfg.data.has_mu:
        for m_idx, cur_mu in enumerate(mu_data):
            if m_idx in cfg.test.test_idx:
                run_test(cfg, apply_fn, opt_params,
                         x_data[m_idx], cur_mu, key, label=m_idx)
    else:
        cur_mu = None
        run_test(cfg, apply_fn, opt_params, x_data, cur_mu, key)

    final_metric = average_metrics()

    save_results(R.RESULT, cfg)

    return final_metric


def build(cfg: Config):

    key = setup(cfg)
    key, net_key, d_key, p_key = jax.random.split(key, num=4)

    x_data, t_data, mu_data = get_dataset(cfg, d_key)

    if cfg.data.has_mu:
        lhs_data, moments = [], []
        for xt_m in x_data:
            mm, lhs, phi_data, sigma_t = get_phi_params(
                cfg, xt_m, t_data, p_key)
            moments.append(mm)
            lhs_data.append(lhs)
        moments, lhs_data = np.asarray(moments), np.asarray(lhs_data)
    else:
        moments, lhs_data, phi_data, sigma_t = get_phi_params(
            cfg, x_data, t_data, p_key)

    dataloader = get_dataloader(
        cfg, x_data, phi_data, lhs_data, t_data, mu_data, sigma_t)

    net, apply_fn, params_init = get_network(cfg, dataloader, net_key)

    loss_fn = get_loss_fn(cfg, apply_fn)

    return key, x_data, dataloader, moments, lhs_data, phi_data, mu_data, sigma_t, loss_fn, net, params_init, apply_fn


if __name__ == "__main__":
    # dont let tensorflow grab GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['TFDS_DATA_DIR'] = '/scratch/jmb1174/tensorflow_datasets'

    run()
