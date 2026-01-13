
import jax
import numpy as np

import gmfm.io.result as R
from gmfm.config.config import Config
from gmfm.data.adv import get_adv_data
from gmfm.data.lanl import get_lanl_clean
from gmfm.data.lz9 import get_lz9_data
from gmfm.data.turb import get_turb_samples
from gmfm.data.wave import get_wave_random_media
from gmfm.utils.tools import normalize, print_ndarray, print_stats, pshape


def get_dataset(cfg: Config, key):

    problem = cfg.dataset
    sub_x = cfg.data.sub_x
    sub_t = cfg.data.sub_t
    n_samples = cfg.data.n_samples
    if n_samples == -1:
        n_samples = None
    n_t = cfg.data.n_t
    n_x = cfg.data.n_x
    norm_method = cfg.data.norm_method

    skey, key = jax.random.split(key)
    shift, scale = 0.0, 1.0

    if problem == "wave":
        n_t = 64
        n_x = 256
        x_data = get_wave_random_media(
            n_samples, n_t, n_x, key, batch_size=32, sigma=cfg.data.alpha
        )
        x_data = x_data[:, ::sub_t, ::sub_x, ::sub_x, None]

    elif problem == "adv":

        x_data = get_adv_data(n_samples, sub_t, sub_x)

    elif problem == "lanl":
        x_data = get_lanl_clean(sub_x, sub_t)
        x_data = x_data[..., None]

    elif problem == "lz9":
        t_eval = np.linspace(0, 20.0, n_t)
        x_data = get_lz9_data(n_samples, t_eval, skey)
    elif problem == "turb":
        x_data = get_turb_samples(n_samples, only_vort=True)

    if cfg.data.normalize:
        x_data, (shift, scale) = normalize(x_data, method=norm_method, axis=-1)
        print_ndarray(shift, title='stats')
        print_ndarray(scale, title='stats')

    N, T = x_data.shape[:2]
    t_data = np.linspace(0, 1, T)
    R.RESULT["normalize_values"] = (shift, scale)

    pshape(x_data)
    print_stats(x_data)

    return x_data, t_data


def extract_well_params(well_data):
    params = []
    for item in well_data:
        p = item['constant_scalars']
        params.append(p.detach().cpu().numpy())
    params = np.asarray(params)
