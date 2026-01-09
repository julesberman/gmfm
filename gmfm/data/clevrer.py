import h5py
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def get_clevrer_data(
    split,
    sub_n,
    sub_x,
    sub_t,
):
    dir = Path("/scratch/jmb1174/clevrer/h5_full/")
    file_path = f"{split}.h5"

    with h5py.File(dir / file_path, "r") as h5_file:
        if sub_n <= 0:
            data = h5_file["clevrer"][:, ::sub_t, ::sub_x, ::sub_x]
        else:
            data = h5_file["clevrer"][:sub_n, ::sub_t, ::sub_x, ::sub_x]

    data = np.asarray(data)

    return data
