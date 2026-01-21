import os
import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import make_smoothing_spline
from gmfm.utils.tools import get_cpu_count

# Prevent BLAS oversubscription inside each process
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_T = None
_MU = None
_BLOCK = None


def _init_worker(t_data, mu, block_size):
    global _T, _MU, _BLOCK
    _T = t_data
    _MU = mu
    _BLOCK = block_size


def _fit_block(block_idx: int):
    j0 = block_idx * _BLOCK
    j1 = min(_MU.shape[1], j0 + _BLOCK)
    Yb = _MU[:, j0:j1]

    spl = make_smoothing_spline(_T, Yb, lam=None, axis=0)
    Db = spl(_T, nu=1)
    print("done", block_idx)
    return block_idx, Db


def _gpu_burn(stop_event, N=8192):
    # IMPORTANT: import JAX inside the spawned process
    import jax
    import jax.numpy as jnp

    a = jnp.ones((N, N), dtype=jnp.float16)
    b = jnp.ones((N, N), dtype=jnp.float16)
    f = jax.jit(lambda x, y: x @ y)

    # warmup/compile
    f(a, b).block_until_ready()

    while not stop_event.is_set():
        f(a, b).block_until_ready()


def get_auto_spline(t_data, mu, block_size=1_000, n_jobs=None, burn_gpu=True, gpu_matmul_n=8192):
    m = mu.shape[1]
    n_blocks = (m + block_size - 1) // block_size
    if n_jobs is None:
        n_jobs = get_cpu_count() or 1

    print("n_jobs", n_jobs)

    gpu_proc = None
    stop_event = None

    # Start GPU burner in a spawned process (safer with JAX/CUDA)
    if burn_gpu:
        spawn_ctx = mp.get_context("spawn")
        stop_event = spawn_ctx.Event()
        gpu_proc = spawn_ctx.Process(target=_gpu_burn, args=(
            stop_event, gpu_matmul_n), daemon=True)
        gpu_proc.start()

    # CPU pool on fork for copy-on-write sharing of mu (Linux)
    fork_ctx = mp.get_context("fork")
    try:
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=fork_ctx,
            initializer=_init_worker,
            initargs=(t_data, mu, block_size),
        ) as ex:
            results = list(ex.map(_fit_block, range(n_blocks)))
    finally:
        if burn_gpu and gpu_proc is not None:
            stop_event.set()
            gpu_proc.join(timeout=2.0)
            if gpu_proc.is_alive():
                gpu_proc.terminate()

    results.sort(key=lambda x: x[0])
    return np.concatenate([Db for _, Db in results], axis=1)
