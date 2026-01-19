
import numpy as np
import pickle
import h5py


def get_hoam_data(path):

    with open(path, "rb") as f:
        obj = pickle.load(f)

    data = obj['sols']
    mu = obj['mu']
    return data, mu


def get_v6_data(n_samples):

    mus = []
    sols = []
    for i in range(0, 11):
        with h5py.File(f"/scratch/jmb1174/data/strongLandauDampingCV{i:02d}.hdf5", "r") as file:
            t_grid = file["t_grid"][:]
            sol = file["sol"][:, 0:n_samples, :]
            mu = file["mu"][()]
        mus.append(mu)
        sols.append(sol)
    sols = np.asarray(sols)
    mus = np.asarray(mus)
    idx = np.argsort(mus)
    mus = mus[idx]
    sols = sols[idx]
    # train_idx = np.asarray([0, 1, 2, 3, 4, 6, 7, 9, 10])
    # test_idx = np.asarray([5])
    t_eval = t_grid
    sols = sols[:, :-1]  # bug sol is too big
    T = int(sols.shape[1] // 6)
    sols = sols[:, :T]
    t_eval = t_eval[:T]

    return sols, mus
