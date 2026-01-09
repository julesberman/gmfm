import glob
import inspect
import os
import random
import string
from functools import partial
from time import time
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from tqdm.auto import tqdm


def jax_key_to_np(key):
    a, b = jax.random.key_data(key)
    seed = (int(a) << 32) | int(b)
    return np.random.default_rng(seed)


def load_with_pattern(directory, filename_pattern):
    search_pattern = os.path.join(directory, filename_pattern)
    matching_files = glob.glob(search_pattern)
    return matching_files


def randkey():
    return jax.random.PRNGKey(random.randint(-1e12, 1e12))


def randkeys(num):
    k = jax.random.PRNGKey(random.randint(-1e12, 1e12))
    return jax.random.split(k, num=num)


def unique_id(n) -> str:
    """creates unique alphanumeric id w/ low collision probability"""
    chars = string.ascii_letters + string.digits  # 64 choices
    id_str = "".join(random.choice(chars) for _ in range(n))
    return id_str


def epoch_time(decimals=0) -> int:
    return int(time() * (10 ** (decimals)))


def pts_array_from_space(space):
    m_grids = jnp.meshgrid(*space, indexing="ij")
    x_pts = jnp.asarray([m.flatten() for m in m_grids]).T
    return x_pts


def make_var_namer():
    """
    Returns a function get_var_name(value) that resolves `value` to the
    caller-of-the-caller's local variable name(s).

    Called from inside pshape():
      make_var_namer -> pshape -> pshape_caller
    so we go back 2 frames.
    """
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back.f_back  # pshape's caller
        local_vars = caller_frame.f_locals

        # Build id(value) -> [names] once
        value_to_names = {}
        for name, v in local_vars.items():
            value_to_names.setdefault(id(v), []).append(name)

        def get_var_name(value: Any) -> str:
            return ", ".join(value_to_names.get(id(value), ["unknown"]))

        return get_var_name
    finally:
        # avoid reference cycles
        del frame


def pshape(*args, title: str = ""):
    get_var_name = make_var_namer()
    dlim = " | "
    if title:
        print(title, end=" || ")

    for arg in args:
        var_name_str = get_var_name(arg)
        if hasattr(arg, "shape"):
            print(f"{var_name_str}: {arg.shape}", end=dlim)
        else:
            print(f"{var_name_str}: no_shape", end=dlim)
    print()


def print_stats(x):
    v = make_var_namer()(x)
    print(
        f"{v} | shape: {x.shape} min: {x.min():.5f}, max: {x.max():.5f}, mean: {x.mean():.5f}, std: {x.std():.5f}, dtype: {x.dtype}, type: {type(x)}")


def get_rand_idx(key, N, bs):
    if bs > N:
        bs = N
    idx = jnp.arange(0, N)
    return jax.random.choice(key, idx, shape=(bs,), replace=False)


def meanvmap(f, in_axes=(0), mean_axes=(0,)):
    return lambda *fargs, **fkwargs: jnp.mean(
        vmap(f, in_axes=in_axes)(*fargs, **fkwargs), axis=mean_axes
    )


def tracewrap(f, axis1=0, axis2=1):
    return lambda *fargs, **fkwargs: jnp.trace(
        f(*fargs, **fkwargs), axis1=axis1, axis2=axis2
    )


def normwrap(f, axis=None, flatten=False):
    if flatten:
        return lambda *fargs, **fkwargs: jnp.linalg.norm(f(*fargs, **fkwargs).reshape(-1))
    else:
        return lambda *fargs, **fkwargs: jnp.linalg.norm(f(*fargs, **fkwargs), axis=axis)


def batchvmap(f, batch_size, in_arg=0, batch_dim=0, pbar=False, to_numpy=False, mean=False):

    def wrap(*fargs, **fkwarg):
        fargs = list(fargs)
        X = fargs[in_arg]
        n_batches = jnp.ceil(X.shape[batch_dim] // batch_size).astype(int)
        n_batches = max(1, n_batches)
        batches = jnp.array_split(X, n_batches, axis=batch_dim)

        in_axes = [None] * len(fargs)
        in_axes[in_arg] = batch_dim
        v_f = vmap(f, in_axes=in_axes)
        result = []
        if mean:
            result = 0.0
        if pbar:
            batches = tqdm(batches)
        for B in batches:
            fargs[in_arg] = B
            a = v_f(*fargs, **fkwarg)
            if mean:
                result += a.mean(axis=0)
            else:
                if to_numpy:
                    a = np.asarray(a)
                result.append(a)

        if mean:
            return result / n_batches

        if to_numpy:
            return np.concatenate(result)

        return jnp.concatenate(result)

    return wrap


def sqwrap(f):
    return lambda *fargs, **fkwargs: jnp.squeeze(f(*fargs, **fkwargs))


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def normalize(x, method="-11", axis=None, eps=1e-8):
    """
    Normalize `x` using different conventions and return reusable stats.

    Parameters
    ----------
    x : array_like
        Input array.
    method : {"zscore", "std", "minmax", "01", "sym", "-11"}, default "zscore"
        - "zscore" / "std": (x - mean) / std
        - "minmax" / "01":  map min->0, max->1
        - "sym" / "-11":    map min->-1, max->1
    axis : int or tuple of int, optional
        Axis/axes to *keep* (i.e., along which shift/scale vary). Statistics are
        computed over all other axes. If None, use all elements (scalar shift/scale).

        Example: if `x` is (N, T, D) and `axis=-1` (or 2), you get D shifts/scales.
    eps : float, default 1e-8
        Small value to avoid division by zero when scale is very small.

    Returns
    -------
    x_norm : ndarray
        Normalized array.
    (shift, scale) : tuple of ndarray
        Shift and scale used. These are returned in broadcast-ready shape so that
        you can directly invert with `unnormalize`.

    Notes
    -----
    To invert the normalization:
        x = unnormalize(x_norm, (shift, scale))
    """
    x = np.asarray(x)
    method = method.lower()

    # axis = axes to KEEP (stats vary along these); reduce over all others
    if axis is None:
        red = None
        keep = ()
    else:
        keep = (axis,) if isinstance(axis, int) else tuple(axis)
        keep = tuple(sorted({a % x.ndim for a in keep}))
        red = tuple(i for i in range(x.ndim) if i not in keep)

    if method in {"zscore", "std"}:
        shift = x.mean(axis=red)
        scale = x.std(axis=red)
    elif method in {"minmax", "01"}:
        mn = x.min(axis=red)
        mx = x.max(axis=red)
        shift, scale = mn, (mx - mn)
    elif method in {"sym", "-11"}:
        mn = x.min(axis=red)
        mx = x.max(axis=red)
        shift, scale = (mx + mn) / 2.0, (mx - mn) / 2.0
    else:
        raise ValueError(f"Unknown method: {method!r}")

    scale = np.where(np.abs(scale) < eps, 1.0, scale)

    # reshape compact stats to broadcast against x
    if axis is not None:
        shape = [1] * x.ndim
        for i, ax in enumerate(keep):
            shape[ax] = shift.shape[i]
        shift = shift.reshape(shape)
        scale = scale.reshape(shape)

    return (x - shift) / scale, (shift, scale)


def unnormalize(x_norm, stats):
    """
    Invert `normalize` given the (shift, scale) stats it returned.

    Parameters
    ----------
    x_norm : array_like
        Normalized array.
    stats : (shift, scale)
        Tuple returned by `normalize` (broadcast-ready).

    Returns
    -------
    x : ndarray
        Unnormalized array in the original scale.
    """
    shift, scale = stats
    return np.asarray(x_norm) * scale + shift


def fold_in_data(*args):
    s = 0.0
    for a in args:
        s += jnp.cos(jnp.linalg.norm(a))
    s *= 1e6
    s = s.astype(jnp.int32)
    return s


def combine_keys(df, n_k, k_arr):
    df[n_k] = df[k_arr].agg(lambda x: "~".join(x.astype(str)), axis=1)
    return df


def jacrand(rng_key, fun, argnums=0):
    """
    Like jacrev, but instead of returning a full Jacobian, it returns a
    single Jacobian-vector product in a random direction (of norm 1).

    Args:
        fun:      Function whose output we care about. Signature:
                  fun(*args, **kwargs) -> output pytree
        argnums:  int or tuple of ints, specifying which positional args
                  to differentiate against. (Default = 0)
        rng_key:  Optional PRNGKey to control randomness. If None,
                  defaults to jax.random.PRNGKey(0). Internally stored and
                  split each time you call the transformed function.

    Returns:
        A new function with the same signature as `fun` that returns:
            ( fun(*args, **kwargs), jvp_value )
        where jvp_value is the result of J_fun(*args, **kwargs) @ r_unit,
        and r_unit is a new random direction (norm 1) each call.
    """

    # Convert argnums to a tuple for uniform handling
    if isinstance(argnums, int):
        argnums_tuple = (argnums,)
    else:
        argnums_tuple = tuple(argnums)

    def wrapper(*args, **kwargs):
        # Extract the "differentiable" arguments (the ones in argnums)
        # as the 'primals' for jvp
        primals = tuple(args[i] for i in argnums_tuple)

        # Build a partial version of fun that only expects the
        # differentiable arguments, while capturing the rest by closure.
        def partial_fun(*dyn_args):
            # Rebuild the entire argument list
            full_args = list(args)
            for idx, val in zip(argnums_tuple, dyn_args, strict=False):
                full_args[idx] = val
            # Call original function
            return fun(*full_args, **kwargs)

        subkeys = jax.random.split(rng_key, len(argnums_tuple))

        tangent_subset = []
        for p, sk in zip(primals, subkeys, strict=False):
            r = jax.random.normal(sk, shape=p.shape)
            # Protect against zero-norm corner cases
            norm = jnp.linalg.norm(r)
            r_unit = r / (norm + 1e-12)
            tangent_subset.append(r_unit)

        tangent_subset = tuple(tangent_subset)

        # Get (primal_out, jvp_out)
        y, jvp = jax.jvp(partial_fun, primals, tangent_subset)

        return jvp

    return wrapper


def get_cpu_count() -> int:
    cpu_count = None
    if hasattr(os, "sched_getaffinity"):
        try:
            cpu_count = len(os.sched_getaffinity(0))
            return cpu_count
        except Exception:
            pass

    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return cpu_count

    try:
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
    except Exception:
        pass

    print("could not get cpu count, returning 1")

    return 1


def get_available_ram_gb():
    paths = [
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",            # older cgroup v1
        "/sys/fs/cgroup/memory.max",                              # cgroup v2
    ]

    for path in paths:
        try:
            with open(path, "r") as f:
                limit = int(f.read().strip())
                if limit > 0 and limit < 1 << 60:  # filter out "no limit" sentinel values
                    return limit / (1024**3)
        except FileNotFoundError:
            continue
    return -1


def hutch_div(f, argnum: int = 1, n_samples: int = 1):
    """
    Hutchinson divergence estimator for:
      - f: R^d -> R^d           (x.shape = (d,))
      - f: R^{b,d} -> R^{b,d}   (x.shape = (b,d))

    Returns div(*args, key, return_fwd=False) whose output is:
      - div_est                    if return_fwd=False
      - (div_est, f_x)             if return_fwd=True

    Shapes:
      - div_est: () or (b,)
      - f_x:     (d,) or (b, d)
    """

    @partial(jax.jit, static_argnames=("return_fwd",))
    def div(*args, key, return_fwd: bool = False):
        x = args[argnum]

        def g(x_arg):
            new_args = list(args)
            new_args[argnum] = x_arg
            return f(*new_args)

        keys = jax.random.split(key, n_samples)

        # Compute primal once, and get a cached linear map v ↦ J(x)v
        f_x, lin = jax.linearize(g, x)

        def one(k):
            v = jax.random.rademacher(k, shape=x.shape, dtype=x.dtype)
            jvp_val = lin(v)  # = J(x) v
            return jnp.sum(v * jvp_val, axis=-1)  # () or (b,)

        div_est = jax.vmap(one)(keys).mean(axis=0)

        return (div_est, f_x) if return_fwd else div_est

    return div


def print_ndarray(
    x: Any,
    n_c: int = 4,
    n_per_line: int = 12,
    scientific: bool = False,
    title: Optional[str] = None,
) -> None:
    """
    Pretty-print any NumPy/JAX array as a flattened 1D list.

    - Always flattens/reshapes to 1D.
    - Tight, clean formatting.
    - Float formatting:
        * scientific=False -> significant digits via 'g' (compact)
        * scientific=True  -> scientific notation via 'e'

    Args:
        x: numpy.ndarray, jax.Array, list, etc.
        n_c: digits for float formatting (precision).
        n_per_line: values per line before wrapping.
        scientific: force scientific notation for floats/complex components.
        title: if provided, printed on the same line before the array.
    """
    a = np.asarray(
        x).reshape(-1)  # works for numpy + jax (copies to host if needed)
    n = a.size

    prefix = f"{title} | " if title else ""
    v_name = make_var_namer()(x)
    prefix = f"{prefix}{v_name}: "

    if n == 0:
        print(prefix + "[]")
        return

    kind = a.dtype.kind  # b,i,u,f,c,O,S,U,...

    if kind == "f":
        spec = "e" if scientific else "g"
        fmt = f"{{:.{n_c}{spec}}}".format
        def to_s(v): return fmt(float(v))
    elif kind == "c":
        spec = "e" if scientific else "g"
        fmt = f"{{:.{n_c}{spec}}}".format

        def to_s(v):
            v = complex(v)
            re = fmt(v.real)
            im_abs = fmt(abs(v.imag))
            sign = "+" if v.imag >= 0 else "-"
            return f"{re}{sign}{im_abs}j"
    elif kind in ("i", "u"):
        def to_s(v): return str(int(v))
    elif kind == "b":
        def to_s(v): return "True" if bool(v) else "False"
    else:
        def to_s(v): return repr(v)

    parts = [to_s(v) for v in a.tolist()]

    if n <= n_per_line:
        print(prefix + "[" + " ".join(parts) + "]")
        return

    lines = []
    for i in range(0, n, n_per_line):
        lines.append("  " + " ".join(parts[i: i + n_per_line]))

    # Title on same line before the opening bracket; array continues nicely on new lines.
    print(prefix + "[\n" + "\n".join(lines) + "\n]")
