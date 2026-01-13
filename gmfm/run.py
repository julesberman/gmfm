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


@hydra.main(version_base=None, config_name="default")
def run(cfg: Config) -> None:

    key, x_data, dataloader, mu, lhs_data, phi_data, sigma_t, loss_fn, net, params_init, apply_fn = build(
        cfg)

    key, key_G, key_test, key_opt = jax.random.split(key, num=4)

    opt_params = train_model(cfg, dataloader,
                             loss_fn, params_init, key_opt, has_aux=True)

    run_test(cfg, apply_fn, opt_params, x_data, key)

    save_results(R.RESULT, cfg)


def build(cfg: Config):

    key = setup(cfg)
    key, net_key, d_key, p_key = jax.random.split(key, num=4)

    x_data, t_data = get_dataset(cfg, d_key)
    mu, lhs_data, phi_data, sigma_t = get_phi_params(
        cfg, x_data, t_data, p_key)
    dataloader = get_dataloader(
        cfg, x_data, phi_data, lhs_data, t_data, sigma_t)

    net, apply_fn, params_init = get_network(cfg, dataloader, net_key)

    loss_fn = get_loss_fn(cfg, apply_fn)

    return key, x_data, dataloader, mu, lhs_data, phi_data, sigma_t, loss_fn, net, params_init, apply_fn


if __name__ == "__main__":
    # dont let tensorflow grab GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['TFDS_DATA_DIR'] = '/scratch/jmb1174/tensorflow_datasets'

    run()
