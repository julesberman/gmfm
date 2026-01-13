from gmfm.config.config import Config
from gmfm.loss.gm import make_gmfm_loss, make_gmfm_loss_resample


def get_loss_fn(cfg: Config, apply_fn):

    if cfg.loss.resample:
        return make_gmfm_loss_resample(cfg, apply_fn)
    else:
        return make_gmfm_loss(cfg, apply_fn)
