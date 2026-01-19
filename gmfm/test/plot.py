
import numpy as np

from gmfm.config.config import Config, get_outpath
from gmfm.utils.plot import plot_grid_movie, scatter_movie


def get_hist_single(frame, nx):
    frame = frame.T
    H, x, y = np.histogram2d(
        frame[0], frame[1], bins=nx, range=[[-1, 1], [-1, 1]])
    return H.T


def get_hist(frame, nx=135):
    hs = []
    for f in frame:
        h = get_hist_single(f, nx)
        hs.append(h)
    return np.asarray(hs)


def plot_sde(cfg: Config, test_sol, true_sol, label=''):
    true_sol = np.nan_to_num(true_sol)
    test_sol = np.nan_to_num(test_sol)

    frames = 75
    out_path = get_outpath()

    true_sol = np.swapaxes(true_sol, 0, 1)
    test_sol = np.swapaxes(test_sol, 0, 1)
    # N = test_sol.shape[1]

    # plot scatter
    try:
        if true_sol.shape[-1] > 2:
            dim_idx = np.asarray([0, 3])
            true_sol_sub = true_sol[:, :, dim_idx]
            test_sol_sub = test_sol[:, :, dim_idx]
        else:
            test_sol_sub = test_sol
            true_sol_sub = true_sol

        plot_sol = np.asarray([true_sol_sub[:, :], test_sol_sub[:, :]])

        scatter_movie(plot_sol, alpha=0.3, xlim=[-1, 1], ylim=[-1, 1],
                      show=False, frames=frames, save_to=f'{out_path}/sol_{label}.gif')
    except Exception as e:
        print(e, "could not plot particles")

    try:
        if true_sol.shape[-1] > 2:
            dim_idx = np.asarray([0, 3])
            true_sol_sub = true_sol[:, :, dim_idx]
            test_sol_sub = test_sol[:, :, dim_idx]
        else:
            test_sol_sub = test_sol
            true_sol_sub = true_sol

        idx_time = np.linspace(0, len(test_sol)-1, frames, dtype=np.int32)
        hist_sol_test = get_hist(test_sol_sub[idx_time])
        hist_sol_true = get_hist(true_sol_sub[idx_time])
        plot_grid_movie([hist_sol_true, hist_sol_test], frames=frames, show=False,
                        save_to=f'{out_path}/hist_{label}.gif', titles_x=['True', 'Pred'], live_cbar=True)
    except Exception as e:
        print(e, "could not plot hist")


def plot_spde(cfg: Config, pred, true, label=""):

    out_path = get_outpath()

    pred = np.nan_to_num(pred)
    true = np.nan_to_num(true)

    n = cfg.test.n_plot
    pred, true = pred[:n], true[:n]
    n, t, h, w, n_channels = true.shape

    ratio = h / w

    fig_size = (8, 8 * ratio)
    c = 0
    plot_pred = pred[..., c]
    to_int = False

    plot_grid_movie(
        plot_pred,
        save_to=f"{out_path}/samples_pred_{label}.gif",
        seconds=5,
        writer="imagemagick",
        aspect="equal",
        cmap="viridis",
        fig_size=fig_size,
        frames=42,
        to_int=to_int,
    )

    for c in range(n_channels):
        sols = np.concatenate([true[:4, ..., c], pred[:4, ..., c]])
        plot_grid_movie(
            sols,
            fig_size=fig_size,
            save_to=f"{out_path}/compare_{label}_c{c}.gif",
            seconds=4,
            writer="imagemagick",
            titles_y=["Pred", "True"],
            grid_height=2,
            grid_width=4,
            aspect="equal",
            cmap="viridis",
            frames=42,
        )
