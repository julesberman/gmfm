from gmfm.config.config import Config
from gmfm.loss.gm import make_gmfm_loss


def get_loss_fn(cfg: Config, apply_fn):

    return make_gmfm_loss(cfg, apply_fn)
