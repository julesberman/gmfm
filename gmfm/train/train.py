import jax
import jax.numpy as jnp
import numpy as np
import optax
import optax.contrib
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm.auto import tqdm

import gmfm.io.result as R
from gmfm.config.config import Config

str_to_opt = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "sgd": optax.sgd,
    "lbfgs": optax.lbfgs,
    "muon": optax.contrib.muon,
}

# ---------- multi-device helpers ----------
_N_DEVICES = jax.local_device_count()
_MULTI_DEVICE = _N_DEVICES > 1


def train_model(cfg: Config, dataloader, loss, params_init, key_opt, has_aux=False, name=''):
    """
    Drop-in replacement: EMA removed entirely.
    Returns optimized params.
    """
    opt_cfg = cfg.optimizer
    opt_params, loss_history = run_train(
        params_init=params_init,
        dataloader=dataloader,
        fwd_fn=loss,
        iters=opt_cfg.iters,
        optimizer=opt_cfg.optimizer,
        learning_rate=opt_cfg.lr,
        scheduler=opt_cfg.scheduler,
        rng=key_opt,
        has_aux=has_aux,
        grad_clip_norm=opt_cfg.grad_clip or None,
        pbar_delay=opt_cfg.pbar_delay,
    )

    R.RESULT["opt_params"] = opt_params
    R.RESULT["loss_history"] = loss_history
    if len(loss_history) > 0:
        R.RESULT[f"{name}_final_loss"] = float(loss_history[-1])

    return opt_params


def run_train(
    params_init,
    dataloader,
    fwd_fn,
    iters: int,
    optimizer: str = "adamw",
    learning_rate: float = 1e-3,
    scheduler: str | None = "cos",
    N: int = 2048,
    rng=None,
    has_aux: bool = False,
    grad_clip_norm: float | None = None,
    pbar_delay: int | None = None,
):
    if rng is None:
        rng = jax.random.PRNGKey(1)

    # ---- LR schedule ----
    if scheduler is not None:
        if scheduler == "cos":
            learning_rate = optax.cosine_decay_schedule(
                init_value=learning_rate, decay_steps=iters, alpha=1e-3
            )
        elif scheduler == "const":
            learning_rate = optax.constant_schedule(value=learning_rate)
        elif scheduler == "linear":
            learning_rate = optax.linear_schedule(
                init_value=learning_rate,
                end_value=learning_rate * 1e-3,
                transition_steps=iters,
            )
        elif scheduler == "warmup":
            learning_rate = optax.warmup_constant_schedule(
                init_value=1e-6, peak_value=learning_rate, warmup_steps=10_000
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

    # ---- Optimizer (optionally with grad clipping) ----
    if optimizer not in str_to_opt:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. Expected one of {list(str_to_opt.keys())}")

    base_tx = str_to_opt[optimizer](learning_rate=learning_rate)
    if grad_clip_norm is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(float(grad_clip_norm)),
            base_tx,
        )
    else:
        tx = base_tx

    opt_state = tx.init(params_init)

    # ---- Single step ----
    def train_step(params, step_rng, opt_state, args):
        def _loss_fn(p):
            return fwd_fn(p, *args, step_rng)

        loss_out, grads = jax.value_and_grad(_loss_fn, has_aux=has_aux)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_out

    # Donate only unique buffers (params, opt_state). No EMA means no alias risk.
    train_step = jax.jit(
        train_step,
        donate_argnums=(0, 2),  # params, opt_state
    )

    # ---- Sharding setup ----
    if _MULTI_DEVICE:
        mesh = jax.make_mesh((_N_DEVICES,), ("batch",))
        SHARD_BATCH = NamedSharding(mesh, P("batch"))
        SHARD_REPL = NamedSharding(mesh, P())
    else:
        mesh = SHARD_BATCH = SHARD_REPL = None

    def jax_put(x, sharding=None, float_16: bool = False):
        if float_16:
            x = x.astype(jnp.bfloat16)
        return jax.device_put(x, sharding) if sharding is not None else jax.device_put(x)

    params = jax_put(params_init, SHARD_REPL)
    opt_state = jax_put(opt_state, SHARD_REPL)

    # ---- Loop ----
    loss_history = []
    interval = max(1, iters // N)

    pbar = tqdm(range(iters), colour="#1DA949")
    dl_iter = iter(dataloader)

    p_inc = 0
    loss_value = None  # assigned after first step

    for step in pbar:
        args = next(dl_iter)
        # Ensure args is a list/tuple pytree of arrays
        args = [jax_put(x, SHARD_BATCH) for x in args]

        rng, step_rng = jax.random.split(rng)

        params, opt_state, loss_out = train_step(
            params, step_rng, opt_state, args)

        if pbar_delay is not None:
            p_inc = (step % pbar_delay)
        else:
            p_inc = 0

        if p_inc == 0:
            if has_aux:
                loss_value, aux = loss_out
                FIELD_WIDTH = 8
                segments = [f"loss: {loss_value:10.4f}".ljust(FIELD_WIDTH)]
                for k, v in aux.items():
                    segments.append(f"{k}: {v:10.4f}".ljust(FIELD_WIDTH))
                pbar.set_description(" | ".join(segments), refresh=False)
            else:
                loss_value = loss_out
                pbar.set_description(f"loss: {loss_value:.6f}", refresh=False)

        if (step % interval) == 0 and loss_value is not None:
            # Move scalar to host for logging history
            loss_history.append(jax.device_get(loss_value).astype(np.float32))

    loss_history = np.array(loss_history, dtype=np.float32)
    return params, loss_history
