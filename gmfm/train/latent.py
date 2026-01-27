"""
Single-file (JAX/Flax/Optax) convolutional VAE-style autoencoder for turbulence data.

Key additions vs a plain AE:
- Variational latent (mu, logvar) + beta-KL regularization (β-VAE) for a smoother, more structured latent space
- KL warmup schedule (best-practice) to avoid early posterior collapse and improve recon stability
- Explicit latent normalization (per-dimension mean/std) computed after training:
    encode_data() returns normalized (optionally squashed) latents
    decode_data() automatically inverts the normalization before decoding

Expected input shape:
  (N, T, 128, 128, 1)  or  (B, 128, 128, 1)

Public API:
  train_autoencoder(data) -> opt_params
  encode_data(data) -> latents_norm
  decode_data(latents_norm) -> decoded_data

Dependencies: numpy, jax, flax, optax, tqdm
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
import optax


# =========================
# Hyperparameters (edit here)
# =========================

AE_CONFIG: Dict[str, Any] = dict(
    # Model
    latent_dim=256,           # unused in spatial-latent model, kept for compatibility
    base_channels=32,         # good default for 128x128
    num_down=3,              # 128 -> 64 -> 32 -> 16
    dropout=0.05,            # small dropout to prevent trivial memorization
    gn_groups=8,             # GroupNorm groups (must divide channels)

    # VAE latent structure
    beta_kl=0.001,          # β-VAE coefficient (1.0 ~ standard VAE)
    kl_warmup_steps=5000,    # ramp beta from 0 to beta_kl over these steps
    # optional free-nats per latent dim (0.0 disables). Try 0.25 if collapse.
    kl_free_nats=0.0,

    # Latent output normalization for downstream models
    latent_norm_eps=1e-6,
    # if True: return tanh-squashed latents in [-1, 1]
    latent_squash=True,
    # tanh(z/scale). Larger => less saturation. 2-4 is typical.
    latent_squash_scale=2.0,
    # encode_data uses posterior mean (mu) if True; else samples
    encode_use_mean=True,

    # Training
    seed=0,
    batch_size=32,
    num_epochs=3,
    val_fraction=0.05,
    shuffle=True,
    drop_last=True,          # keep batch shapes stable for JIT

    # Optimizer / schedule
    learning_rate=5e-4,
    end_lr_ratio=0.05,       # end_lr = lr * end_lr_ratio
    warmup_steps=500,
    weight_decay=1e-4,
    grad_clip=1.0,
    ema_decay=0.999,         # EMA params for eval / encode / decode

    # Reconstruction loss
    mae_weight=0.10,         # MSE + mae_weight*MAE

    # Logging / early stopping
    log_every=100,
    early_stop_patience=8,   # epochs without val improvement
    eval_batches_limit=None,  # e.g. 50 to limit eval cost; None = full val

    # Numerical
    eps=1e-6,
    logvar_clip=(-12.0, 6.0),  # stability guard for logvar
)

# =========================
# Internal global context
# =========================

_AE_CONTEXT: Optional[Dict[str, Any]] = None


# =========================
# Model definition
# =========================

class ConvBlock(nn.Module):
    features: int
    gn_groups: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = nn.Conv(self.features, kernel_size=(3, 3),
                    padding="SAME", use_bias=False)(x)
        x = nn.GroupNorm(num_groups=self.gn_groups, epsilon=1e-5)(x)
        x = nn.swish(x)
        if self.dropout and self.dropout > 0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return x


class Encoder(nn.Module):
    base_channels: int
    num_down: int
    dropout: float
    gn_groups: int
    logvar_clip: Tuple[float, float]

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # x: [B, 128, 128, 1]
        h = nn.Conv(self.base_channels, kernel_size=(3, 3), padding="SAME")(x)

        ch = self.base_channels
        for _ in range(self.num_down):
            h = ConvBlock(ch, self.gn_groups, self.dropout)(h, train)
            h = ConvBlock(ch, self.gn_groups, self.dropout)(h, train)
            ch *= 2
            h = nn.Conv(ch, kernel_size=(4, 4),
                        strides=(2, 2), padding="SAME")(h)

        h = ConvBlock(ch, self.gn_groups, self.dropout)(h, train)

        # Spatial latent: [B, 16, 16, 1]
        mu = nn.Conv(1, kernel_size=(3, 3), padding="SAME")(h)
        logvar = nn.Conv(1, kernel_size=(3, 3), padding="SAME")(h)
        lo, hi = self.logvar_clip
        logvar = jnp.clip(logvar, lo, hi)
        return mu, logvar


class Decoder(nn.Module):
    base_channels: int
    num_down: int
    dropout: float
    gn_groups: int

    @nn.compact
    def __call__(self, z: jnp.ndarray, train: bool) -> jnp.ndarray:
        # z: [B, 16, 16, 1]
        ch = self.base_channels * (2 ** self.num_down)

        # Lift channels at bottleneck resolution while keeping spatial structure.
        h = nn.Conv(ch, kernel_size=(3, 3), padding="SAME")(z)
        h = ConvBlock(ch, self.gn_groups, self.dropout)(h, train)

        for _ in range(self.num_down):
            ch //= 2
            h = nn.ConvTranspose(ch, kernel_size=(
                4, 4), strides=(2, 2), padding="SAME")(h)
            h = ConvBlock(ch, self.gn_groups, self.dropout)(h, train)
            h = ConvBlock(ch, self.gn_groups, self.dropout)(h, train)

        x_hat = nn.Conv(1, kernel_size=(3, 3), padding="SAME")(h)
        return x_hat


class VariationalAutoEncoder(nn.Module):
    latent_dim: int
    base_channels: int
    num_down: int
    dropout: float
    gn_groups: int
    logvar_clip: Tuple[float, float]

    def setup(self):
        # Separate submodules -> separate param scopes -> safe with method=encode/decode
        self.encoder = Encoder(
            base_channels=self.base_channels,
            num_down=self.num_down,
            dropout=self.dropout,
            gn_groups=self.gn_groups,
            logvar_clip=self.logvar_clip,
        )
        self.decoder = Decoder(
            base_channels=self.base_channels,
            num_down=self.num_down,
            dropout=self.dropout,
            gn_groups=self.gn_groups,
        )

    def encode(self, x: jnp.ndarray, train: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.encoder(x, train)

    def decode(self, z: jnp.ndarray, train: bool) -> jnp.ndarray:
        return self.decoder(z, train)

    def __call__(self, x: jnp.ndarray, train: bool, rng: Optional[jnp.ndarray] = None):
        mu, logvar = self.encode(x, train=train)
        if train:
            if rng is None:
                raise ValueError(
                    "rng must be provided for training (reparameterization).")
            eps = jax.random.normal(rng, shape=mu.shape, dtype=mu.dtype)
            z = mu + jnp.exp(0.5 * logvar) * eps
        else:
            z = mu
        x_hat = self.decode(z, train=train)
        return x_hat, mu, logvar, z


# =========================
# Train state
# =========================

@struct.dataclass
class VAETrainState:
    step: int
    params: Any
    opt_state: Any
    ema_params: Any


# =========================
# Data utilities
# =========================

def _flatten_nt(data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int, int]]:
    x = np.asarray(data)

    if x.ndim == 5:
        N, T, H, W, C = x.shape
        flat = x.reshape((N * T, H, W, C))
        return flat, (N, T, H, W, C)

    if x.ndim == 4:
        B, H, W, C = x.shape
        return x, (B, 1, H, W, C)

    if x.ndim == 3:
        H, W, C = x.shape
        return x[None, ...], (1, 1, H, W, C)

    raise ValueError(
        f"Unsupported data shape {x.shape}. Expected 5D, 4D, or 3D array.")


def _standardize_fit(x: np.ndarray, eps: float) -> Dict[str, float]:
    x64 = x.astype(np.float64)
    mean = float(x64.mean())
    std = float(x64.std())
    std = max(std, eps)
    return dict(mean=mean, std=std, data_min=float(x64.min()), data_max=float(x64.max()))


def _standardize_apply(x: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return (x - stats["mean"]) / stats["std"]


def _destandardize_apply(x: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return x * stats["std"] + stats["mean"]


def _make_splits(B: int, val_fraction: float, seed: int, shuffle: bool) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(B)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    n_val = max(1, int(round(B * val_fraction))) if val_fraction > 0 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if train_idx.size == 0:
        train_idx = idx
        val_idx = idx
    return train_idx, val_idx


def _iter_batches(
    x: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool,
    epoch: int,
):
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        idx = indices.copy()
        rng.shuffle(idx)
    else:
        idx = indices

    n = idx.size
    if drop_last:
        n_batches = n // batch_size
    else:
        n_batches = (n + batch_size - 1) // batch_size

    for b in range(n_batches):
        s = b * batch_size
        e = min((b + 1) * batch_size, n)
        yield x[idx[s:e]]


def _pad_to_batch(x: np.ndarray, batch_size: int) -> Tuple[np.ndarray, int]:
    b = x.shape[0]
    if b == batch_size:
        return x, 0
    pad = batch_size - b
    if b == 0:
        raise ValueError("Empty batch encountered.")
    last = x[-1:]
    x_pad = np.concatenate([x, np.repeat(last, pad, axis=0)], axis=0)
    return x_pad, pad


# =========================
# Latent normalization helpers
# =========================

def _np_atanh(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.clip(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * np.log((1.0 + x) / (1.0 - x))


def _jnp_atanh(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    x = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * jnp.log((1.0 + x) / (1.0 - x))


def _normalize_latents(z: np.ndarray, latent_stats: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> np.ndarray:
    mean = latent_stats["mean"]
    std = latent_stats["std"]
    z_w = (z - mean) / std
    if cfg["latent_squash"]:
        s = float(cfg["latent_squash_scale"])
        z_w = np.tanh(z_w / s)
    return z_w.astype(np.float32, copy=False)


def _denormalize_latents(z_norm: np.ndarray, latent_stats: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> np.ndarray:
    z_w = z_norm.astype(np.float32, copy=False)
    if cfg["latent_squash"]:
        s = float(cfg["latent_squash_scale"])
        z_w = _np_atanh(z_w, eps=float(cfg["latent_norm_eps"])) * s
    mean = latent_stats["mean"]
    std = latent_stats["std"]
    z = z_w * std + mean
    return z.astype(np.float32, copy=False)


# =========================
# Metrics (original units)
# =========================

def _metrics_numpy(x_true: np.ndarray, x_pred: np.ndarray, data_range: float, eps: float) -> Dict[str, float]:
    err = x_pred - x_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(max(mse, 0.0)))
    denom = float(np.sum((x_true - x_true.mean()) ** 2) + eps)
    sse = float(np.sum(err ** 2))
    r2 = float(1.0 - (sse / denom))
    dr = max(float(data_range), eps)
    psnr = float(20.0 * math.log10(dr / max(rmse, eps)))
    return dict(mse=mse, mae=mae, rmse=rmse, r2=r2, psnr=psnr)


# =========================
# Training
# =========================

def train_autoencoder(data: np.ndarray, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Train the VAE-style autoencoder and store it in a module-level context for encode/decode calls.
    Returns opt_params containing trained EMA params + data/latent normalization stats.
    """
    global _AE_CONTEXT

    cfg = dict(AE_CONFIG)
    if config is not None:
        cfg.update(config)

    x_flat, orig_shape = _flatten_nt(data)
    if x_flat.shape[1:4] != (128, 128, 1):
        raise ValueError(
            f"Expected spatial shape (128,128,1). Got {x_flat.shape[1:4]}.")

    x_flat = x_flat.astype(np.float32, copy=False)

    # Fit + apply data standardization
    stats = _standardize_fit(x_flat, eps=float(cfg["eps"]))
    x_std = _standardize_apply(x_flat, stats).astype(np.float32, copy=False)

    B = x_std.shape[0]
    train_idx, val_idx = _make_splits(
        B, float(cfg["val_fraction"]), int(cfg["seed"]), bool(cfg["shuffle"]))

    model = VariationalAutoEncoder(
        latent_dim=int(cfg["latent_dim"]),
        base_channels=int(cfg["base_channels"]),
        num_down=int(cfg["num_down"]),
        dropout=float(cfg["dropout"]),
        gn_groups=int(cfg["gn_groups"]),
        logvar_clip=tuple(cfg["logvar_clip"]),
    )

    # Init params
    key = jax.random.PRNGKey(int(cfg["seed"]))
    dummy = jnp.zeros((int(cfg["batch_size"]), 128, 128, 1), dtype=jnp.float32)
    variables = model.init(
        {"params": key, "dropout": key}, dummy, train=True, rng=key)
    params = variables["params"]
    ema_params = params

    # LR schedule: warmup + cosine decay
    steps_per_epoch = max(
        1,
        int(math.floor(train_idx.size / int(cfg["batch_size"]))) if cfg["drop_last"]
        else int(math.ceil(train_idx.size / int(cfg["batch_size"]))),
    )
    total_steps = int(cfg["num_epochs"]) * steps_per_epoch
    decay_steps = max(1, total_steps - int(cfg["warmup_steps"]))

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(cfg["learning_rate"]),
        warmup_steps=int(cfg["warmup_steps"]),
        decay_steps=int(decay_steps),
        end_value=float(cfg["learning_rate"]) * float(cfg["end_lr_ratio"]),
    )

    tx = optax.chain(
        optax.clip_by_global_norm(float(cfg["grad_clip"])),
        optax.adamw(learning_rate=lr_schedule,
                    weight_decay=float(cfg["weight_decay"])),
    )
    opt_state = tx.init(params)
    state = VAETrainState(step=0, params=params,
                          opt_state=opt_state, ema_params=ema_params)

    # VAE settings
    beta_kl = float(cfg["beta_kl"])
    kl_warmup_steps = int(cfg["kl_warmup_steps"])
    free_nats = float(cfg["kl_free_nats"])
    mae_w = float(cfg["mae_weight"])
    ema_decay = float(cfg["ema_decay"])

    def beta_at_step(step):
        # step can be Python int or JAX scalar
        step_f = jnp.asarray(step, dtype=jnp.float32)
        warm = jnp.asarray(kl_warmup_steps, dtype=jnp.float32)

        # avoid Python control flow; all JAX
        frac = jnp.where(warm > 0.0, jnp.minimum(1.0, step_f / warm), 1.0)
        return jnp.asarray(beta_kl, dtype=jnp.float32) * frac

    def loss_and_metrics(params_local, batch, rng_reparam, rng_dropout, step: int):
        # Forward (train=True uses reparam sample)
        if cfg["dropout"] and cfg["dropout"] > 0:
            x_hat, mu, logvar, z = model.apply(
                {"params": params_local},
                batch,
                train=True,
                rng=rng_reparam,
                rngs={"dropout": rng_dropout},
            )
        else:
            x_hat, mu, logvar, z = model.apply(
                {"params": params_local},
                batch,
                train=True,
                rng=rng_reparam,
            )

        mse = jnp.mean((x_hat - batch) ** 2)
        mae = jnp.mean(jnp.abs(x_hat - batch))
        recon = mse + mae_w * mae

        # KL(q(z|x) || N(0,1)) per sample then mean
        # KL = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
        kl_per = 0.5 * jnp.sum(jnp.exp(logvar) + mu **
                               2 - 1.0 - logvar, axis=(1, 2, 3))
        if free_nats > 0.0:
            latent_dims = float(mu.shape[1] * mu.shape[2] * mu.shape[3])
            kl_floor = free_nats * latent_dims
            kl_per = jnp.maximum(kl_per, kl_floor)
        kl = jnp.mean(kl_per)

        b = beta_at_step(step)
        loss = recon + b * kl

        mets = dict(
            loss=loss,
            recon=recon,
            mse=mse,
            mae=mae,
            kl=kl,
            beta=jnp.array(b, dtype=batch.dtype),
            mu_mean=jnp.mean(mu),
            mu_std=jnp.std(mu),
        )
        return loss, mets

    @jax.jit
    def train_step(state_local: VAETrainState, batch: jnp.ndarray, rng: jnp.ndarray):
        rng, r_reparam, r_drop = jax.random.split(rng, 3)

        def f(p):
            return loss_and_metrics(p, batch, r_reparam, r_drop, state_local.step)

        (loss, mets), grads = jax.value_and_grad(
            f, has_aux=True)(state_local.params)
        updates, new_opt_state = tx.update(
            grads, state_local.opt_state, state_local.params)
        new_params = optax.apply_updates(state_local.params, updates)

        new_ema = jax.tree_util.tree_map(
            lambda e, p: ema_decay * e + (1.0 - ema_decay) * p,
            state_local.ema_params,
            new_params,
        )
        new_state = VAETrainState(
            step=state_local.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            ema_params=new_ema,
        )
        return new_state, mets, rng

    @jax.jit
    def eval_step(params_local: Any, batch: jnp.ndarray):
        # Deterministic eval: use posterior mean (mu) as z
        x_hat, mu, logvar, z = model.apply(
            {"params": params_local}, batch, train=False, rng=None)
        mse = jnp.mean((x_hat - batch) ** 2)
        mae = jnp.mean(jnp.abs(x_hat - batch))
        recon = mse + mae_w * mae

        kl_per = 0.5 * jnp.sum(jnp.exp(logvar) + mu **
                               2 - 1.0 - logvar, axis=(1, 2, 3))
        if free_nats > 0.0:
            latent_dims = float(mu.shape[1] * mu.shape[2] * mu.shape[3])
            kl_floor = free_nats * latent_dims
            kl_per = jnp.maximum(kl_per, kl_floor)
        kl = jnp.mean(kl_per)

        b = beta_at_step(int(1e9))  # full beta for reporting
        loss = recon + b * kl

        return dict(loss=loss, recon=recon, mse=mse, mae=mae, kl=kl, mu_mean=jnp.mean(mu), mu_std=jnp.std(mu))

    def run_eval(params_local: Any) -> Dict[str, float]:
        total = {k: 0.0 for k in ["loss", "recon",
                                  "mse", "mae", "kl", "mu_mean", "mu_std"]}
        n_batches = 0
        limit = cfg["eval_batches_limit"]
        bs = int(cfg["batch_size"])

        for i, batch_np in enumerate(_iter_batches(
            x_std, val_idx, batch_size=bs, shuffle=False, seed=int(cfg["seed"]), drop_last=False, epoch=0
        )):
            if limit is not None and i >= int(limit):
                break
            batch_np, pad = _pad_to_batch(batch_np, bs)
            batch = jnp.asarray(batch_np)
            mets = eval_step(params_local, batch)

            real_n = batch_np.shape[0] - pad
            w = float(real_n) / float(batch_np.shape[0])
            for k in total:
                total[k] += float(mets[k]) * w
            n_batches += 1

        if n_batches == 0:
            return {k: float("nan") for k in total}
        for k in total:
            total[k] /= n_batches
        return total

    best_val = float("inf")
    best_ema_params = state.ema_params
    bad_epochs = 0

    pbar = tqdm(total=total_steps, desc="train_autoencoder",
                dynamic_ncols=True)
    running = None

    rng = jax.random.PRNGKey(int(cfg["seed"]) + 12345)

    for epoch in range(int(cfg["num_epochs"])):
        for batch_np in _iter_batches(
            x_std, train_idx,
            batch_size=int(cfg["batch_size"]),
            shuffle=bool(cfg["shuffle"]),
            seed=int(cfg["seed"]),
            drop_last=bool(cfg["drop_last"]),
            epoch=epoch,
        ):
            batch = jnp.asarray(batch_np)
            state, mets, rng = train_step(state, batch, rng)

            mets_host = {k: float(mets[k]) for k in mets}
            if running is None:
                running = mets_host
            else:
                for k in running:
                    running[k] = 0.95 * running[k] + 0.05 * mets_host[k]

            if state.step % int(cfg["log_every"]) == 0:
                lr_now = float(lr_schedule(state.step))
                pbar.set_postfix(
                    loss=f"{running['loss']:.3e}",
                    recon=f"{running['recon']:.3e}",
                    kl=f"{running['kl']:.3e}",
                    beta=f"{running['beta']:.2f}",
                    lr=f"{lr_now:.2e}",
                )
            pbar.update(1)

        val_mets = run_eval(state.ema_params)
        if val_mets["loss"] + 1e-12 < best_val:
            best_val = val_mets["loss"]
            best_ema_params = state.ema_params
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(cfg["early_stop_patience"]):
            break

    pbar.close()

    # After training: compute per-dimension latent stats (using mu) for robust normalization in encode/decode
    latent_stats = _compute_latent_stats(model, best_ema_params, x_std, cfg)

    # Package opt_params + set global context
    opt_params: Dict[str, Any] = dict(
        params=best_ema_params,   # inference uses EMA weights
        config=cfg,
        stats=stats,
        latent_stats=latent_stats,
        orig_shape=orig_shape,
        model_kwargs=dict(
            latent_dim=int(cfg["latent_dim"]),
            base_channels=int(cfg["base_channels"]),
            num_down=int(cfg["num_down"]),
            dropout=float(cfg["dropout"]),
            gn_groups=int(cfg["gn_groups"]),
            logvar_clip=tuple(cfg["logvar_clip"]),
        ),
    )
    _AE_CONTEXT = dict(model=model, opt_params=opt_params)

    # Print rough reconstruction stats (validation, original units) + latent normalization summary
    _print_recon_stats(model, x_std, val_idx, opt_params)

    return opt_params


def _compute_latent_stats(
    model: VariationalAutoEncoder,
    params: Any,
    x_std: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    bs = int(cfg["batch_size"])
    H = 128 // (2 ** int(cfg["num_down"]))
    W = 128 // (2 ** int(cfg["num_down"]))
    C = 1

    sum_mu = np.zeros((H, W, C), dtype=np.float64)
    sum_mu2 = np.zeros((H, W, C), dtype=np.float64)
    count = 0

    @jax.jit
    def enc_mu(params_local: Any, batch: jnp.ndarray):
        mu, logvar = model.apply(
            {"params": params_local},
            batch,
            train=False,
            method=VariationalAutoEncoder.encode,
        )
        return mu

    idx_all = np.arange(x_std.shape[0])
    for batch_np in _iter_batches(x_std, idx_all, batch_size=bs, shuffle=False, seed=0, drop_last=False, epoch=0):
        batch_np, pad = _pad_to_batch(batch_np, bs)
        mu = np.asarray(enc_mu(params, jnp.asarray(batch_np)))
        if pad > 0:
            mu = mu[:-pad]
        sum_mu += mu.sum(axis=0, dtype=np.float64)
        sum_mu2 += (mu * mu).sum(axis=0, dtype=np.float64)
        count += mu.shape[0]

    mean = sum_mu / max(count, 1)
    var = sum_mu2 / max(count, 1) - mean * mean
    var = np.maximum(var, float(cfg["latent_norm_eps"]))
    std = np.sqrt(var)

    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32))


def _print_recon_stats(
    model: VariationalAutoEncoder,
    x_std: np.ndarray,
    val_idx: np.ndarray,
    opt_params: Dict[str, Any],
) -> None:
    cfg = opt_params["config"]
    stats = opt_params["stats"]

    @jax.jit
    def recon_mu(params_local: Any, batch: jnp.ndarray):
        x_hat, mu, logvar, z = model.apply(
            {"params": params_local}, batch, train=False, rng=None)
        return x_hat, mu

    data_range = float(stats["data_max"] - stats["data_min"])
    eps = float(cfg["eps"])
    bs = int(cfg["batch_size"])
    limit = cfg["eval_batches_limit"]

    mse_acc = mae_acc = rmse_acc = r2_acc = psnr_acc = 0.0
    mu_mean_acc = mu_std_acc = 0.0
    nb = 0

    for i, batch_np in enumerate(_iter_batches(
        x_std, val_idx, batch_size=bs, shuffle=False, seed=int(cfg["seed"]), drop_last=False, epoch=0
    )):
        if limit is not None and i >= int(limit):
            break
        batch_np, pad = _pad_to_batch(batch_np, bs)
        x_hat_std, mu = recon_mu(opt_params["params"], jnp.asarray(batch_np))
        x_hat_std = np.asarray(x_hat_std)
        mu = np.asarray(mu)

        if pad > 0:
            x_hat_std = x_hat_std[:-pad]
            batch_np = batch_np[:-pad]
            mu = mu[:-pad]

        x_true = _destandardize_apply(batch_np, stats)
        x_pred = _destandardize_apply(x_hat_std, stats)
        mets = _metrics_numpy(x_true, x_pred, data_range=data_range, eps=eps)

        mse_acc += mets["mse"]
        mae_acc += mets["mae"]
        rmse_acc += mets["rmse"]
        r2_acc += mets["r2"]
        psnr_acc += mets["psnr"]
        mu_mean_acc += float(mu.mean())
        mu_std_acc += float(mu.std())
        nb += 1

    if nb == 0:
        print("Reconstruction stats: (insufficient validation data to report)")
        return

    latent_stats = opt_params["latent_stats"]
    print("\nAutoencoder training complete. Rough reconstruction stats (validation, original units):")
    print(f"  MSE:  {mse_acc/nb:.6e}")
    print(f"  RMSE: {rmse_acc/nb:.6e}")
    print(f"  MAE:  {mae_acc/nb:.6e}")
    print(f"  R^2:  {r2_acc/nb:.6f}")
    print(f"  PSNR: {psnr_acc/nb:.3f} dB (range={data_range:.6e})")
    print("Latent (posterior mean) summary (validation):")
    print(f"  mean(mu): {mu_mean_acc/nb:.6e} | std(mu): {mu_std_acc/nb:.6e}")
    print("Latent normalization (computed over full dataset, mu):")
    print(
        f"  mean(mean_dim): {float(latent_stats['mean'].mean()):.6e} | mean(std_dim): {float(latent_stats['std'].mean()):.6e}")
    if cfg["latent_squash"]:
        print(
            f"  encode_data returns tanh-squashed normalized latents in [-1,1] (scale={cfg['latent_squash_scale']})")
    else:
        print("  encode_data returns normalized (whitened) latents (approximately unit scale).")
    print("")


# =========================
# Encode / Decode APIs
# =========================

def encode_data(data: np.ndarray, opt_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Encode data into *normalized* latents suitable for downstream sequence/trajectory models.

    Returns:
      - If input is (N,T,128,128,1): (N,T,16,16,1)
      - If input is (B,128,128,1):   (B,16,16,1)

    Normalization:
      z = mu (default) or sample
      z_whiten = (z - mean)/std   (mean/std computed after training)
      if latent_squash: z_out = tanh(z_whiten / scale)  in [-1,1]
    """
    ctx = _require_context(opt_params)
    opt_params = ctx["opt_params"]
    cfg = opt_params["config"]
    stats = opt_params["stats"]
    latent_stats = opt_params["latent_stats"]

    model = VariationalAutoEncoder(**opt_params["model_kwargs"])

    x_flat, orig_shape = _flatten_nt(data)
    x_flat = x_flat.astype(np.float32, copy=False)
    x_std = _standardize_apply(x_flat, stats).astype(np.float32, copy=False)

    bs = int(cfg["batch_size"])

    @jax.jit
    def enc_mu_logvar(params_local: Any, batch: jnp.ndarray):
        mu, logvar = model.apply(
            {"params": params_local},
            batch,
            train=False,
            method=VariationalAutoEncoder.encode,
        )
        return mu, logvar

    rng = jax.random.PRNGKey(int(cfg["seed"]) + 999)
    zs = []
    idx_all = np.arange(x_std.shape[0])

    use_mean = bool(cfg.get("encode_use_mean", True))

    for batch_np in _iter_batches(x_std, idx_all, batch_size=bs, shuffle=False, seed=0, drop_last=False, epoch=0):
        batch_np, pad = _pad_to_batch(batch_np, bs)

        mu, logvar = enc_mu_logvar(opt_params["params"], jnp.asarray(batch_np))
        mu = np.asarray(mu)
        logvar = np.asarray(logvar)

        if not use_mean:
            rng, rk = jax.random.split(rng)
            eps = np.asarray(jax.random.normal(
                rk, shape=mu.shape, dtype=jnp.float32))
            z = mu + np.exp(0.5 * logvar) * eps
        else:
            z = mu

        if pad > 0:
            z = z[:-pad]
        zs.append(z)

    z = np.concatenate(zs, axis=0)
    z_norm = _normalize_latents(z, latent_stats, cfg)

    N, T, H, W, C = orig_shape
    if T == 1 and np.asarray(data).ndim == 4:
        return z_norm
    return z_norm.reshape((N, T, z_norm.shape[1], z_norm.shape[2], z_norm.shape[3]))


def decode_data(latents: np.ndarray, opt_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Decode *normalized* latents (i.e., the output of encode_data) back to fields.

    Accepts:
      latents: (N,T,16,16,1) or (B,16,16,1)

    Automatically inverts:
      if latent_squash: z_whiten = atanh(latents) * scale
      z = z_whiten*std + mean
    """
    ctx = _require_context(opt_params)
    opt_params = ctx["opt_params"]
    cfg = opt_params["config"]
    latent_stats = opt_params["latent_stats"]

    model = VariationalAutoEncoder(**opt_params["model_kwargs"])

    z_norm = np.asarray(latents, dtype=np.float32)
    if z_norm.ndim == 5:
        N, T, H, W, C = z_norm.shape
        z_norm_flat = z_norm.reshape((N * T, H, W, C))
        restore_nt = True
    elif z_norm.ndim == 4:
        z_norm_flat = z_norm
        restore_nt = False
        N, T = z_norm.shape[0], 1
    else:
        raise ValueError(
            f"Unsupported latents shape {z_norm.shape}. Expected 5D or 4D array.")

    z_flat = _denormalize_latents(z_norm_flat, latent_stats, cfg)

    bs = int(cfg["batch_size"])

    @jax.jit
    def dec(params_local: Any, batch_z: jnp.ndarray):
        x_hat = model.apply(
            {"params": params_local},
            batch_z,
            train=False,
            method=VariationalAutoEncoder.decode,
        )
        return x_hat

    outs = []
    idx_all = np.arange(z_flat.shape[0])
    for batch_z in _iter_batches(z_flat, idx_all, batch_size=bs, shuffle=False, seed=0, drop_last=False, epoch=0):
        batch_z, pad = _pad_to_batch(batch_z, bs)
        x_hat_std = dec(opt_params["params"], jnp.asarray(batch_z))
        x_hat_std = np.asarray(x_hat_std)
        if pad > 0:
            x_hat_std = x_hat_std[:-pad]
        outs.append(x_hat_std)

    x_hat_std = np.concatenate(outs, axis=0)
    x_hat = _destandardize_apply(
        x_hat_std, opt_params["stats"]).astype(np.float32, copy=False)

    if restore_nt:
        return x_hat.reshape((N, T, 128, 128, 1))
    return x_hat


def _require_context(opt_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    global _AE_CONTEXT
    if opt_params is not None:
        if _AE_CONTEXT is None:
            model = VariationalAutoEncoder(**opt_params["model_kwargs"])
            _AE_CONTEXT = dict(model=model, opt_params=opt_params)
        return dict(model=_AE_CONTEXT["model"], opt_params=opt_params)
    if _AE_CONTEXT is None:
        raise RuntimeError(
            "Autoencoder not trained. Call train_autoencoder(data) first (or pass opt_params).")
    return _AE_CONTEXT


__all__ = ["train_autoencoder", "encode_data", "decode_data", "AE_CONFIG"]
