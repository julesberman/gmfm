from functools import partial

import jax
import numpy as np
from gmfm.config.config import Config, Network
from gmfm.net.mlp import DNN
from gmfm.net.unet import UNet
from gmfm.utils.tools import pshape


def get_arch(net_cfg: Network, out_channels):

    if net_cfg.arch == "unet":
        net = get_unet_size(net_cfg.size, out_channels,
                            net_cfg.emb_features)
    if net_cfg.arch == "mlp":
        net = DNN(width=128, depth=7,
                  out_features=out_channels, residual=net_cfg.residual)
    return net


def get_unet_size(size, out_channels, emb_features, n_classes=1):

    UnetConstructor = partial(UNet, out_channels)

    if size == "s":
        net = UnetConstructor(
            feature_depths=[96, 128],
            emb_features=emb_features,
            num_res_blocks=2,
            num_middle_res_blocks=1,
            n_classes=n_classes
        )
    elif size == "m":
        net = UnetConstructor(
            feature_depths=[128, 128, 360],
            emb_features=emb_features,
            num_res_blocks=2,
            num_middle_res_blocks=2,
            n_classes=n_classes
        )
    elif size == "l":
        net = UnetConstructor(
            feature_depths=[128, 256, 512],
            emb_features=emb_features,
            num_res_blocks=2,
            num_middle_res_blocks=2,
            n_classes=n_classes
        )

    return net


def get_network(cfg: Config, dataloader, key):

    batch = next(iter(dataloader))
    xt_batch, time = batch[:2]
    out_channels = xt_batch.shape[-1]
    time = np.ones((xt_batch.shape[0], 1))

    pshape(*batch, title='dataloader sample')

    net = get_arch(cfg.net, out_channels)
    params_init = net.init(key, xt_batch, time, None)

    def apply_fn(params, xt, t):
        return net.apply(params, xt, t, None)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params_init))
    print(f"n_params {param_count:,}")

    return net, apply_fn, params_init
