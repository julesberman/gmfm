from pathlib import Path

import h5py


def get_lanl_clean(sub_x, sub_t):
    dir = Path("/scratch/jmb1174/lanl_clean.h5")

    with h5py.File(dir, "r") as h5_file:
        sols = h5_file["saturation"][:, ::sub_t, ::sub_x, ::sub_x,]

    print(sols.shape, sols.dtype)

    return sols
