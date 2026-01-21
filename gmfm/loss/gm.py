import jax
import jax.numpy as jnp
from einops import rearrange

from gmfm.config.config import Config
from gmfm.loss.rff import (
    grad_phi_weights,
    make_rff_params_jnp,
    rff_grad_dot_v,
    rff_laplace_phi,
    rff_phi,
)
from gmfm.utils.tools import pshape


def make_gmfm_loss(cfg: Config, apply_fn):

    sigma = cfg.loss.sigma
    reg_kin = cfg.loss.reg_kin
    reg_traj = cfg.loss.reg_traj
    relative = cfg.loss.relative
    normalize = cfg.loss.normalize
    has_mu = cfg.data.has_mu

    grad_fn = rff_grad_dot_v
    lap_fn = rff_laplace_phi

    def loss_fn(params, x_t, t, omegas, lhs, mu, xtp1_batch, dt, key):
        final_loss = 0.0
        aux = {}

        if has_mu:
            v_t = apply_fn(params, x_t, t, mu)
        else:
            v_t = apply_fn(params, x_t, t, None)

        v_t = rearrange(v_t, 'N ... -> N (...)')
        x_t = rearrange(x_t, 'N ... -> N (...)')

        # E[∇phi · v]
        g_t = grad_fn(x_t, v_t, omegas)  # (2M,)

        # E[Δphi]
        if sigma > 0.0:
            lap_t = lap_fn(x_t, omegas)
            rhs = g_t + 0.5 * (sigma**2) * lap_t
        else:
            rhs = g_t

        err2 = (lhs - rhs)**2

        w = None
        if normalize == 'omega':
            w0 = 1.0 / (jnp.sum(omegas**2, axis=-1) + 1e-8)     # (M,)
            w = jnp.concatenate([w0, w0], axis=0)              # (2M,)
            w = jax.lax.stop_gradient(w)

        elif normalize == 'grad':
            w = grad_phi_weights(x_t, omegas, eps=1e-8)        # (2M,)
            w = w / (jnp.mean(w) + 1e-8)                       # optional
            w = jnp.clip(w, 0.0, 1e3)                          # optional
            w = jax.lax.stop_gradient(w)

        if relative:
            if w is None:
                final_loss = jnp.mean(err2) / (jnp.mean(lhs**2) + 1e-8)
            else:
                final_loss = jnp.sum(w * err2) / (jnp.sum(w * lhs**2) + 1e-8)
        else:
            if w is None:
                final_loss = jnp.mean(err2)
            else:
                final_loss = jnp.sum(w * err2) / (jnp.sum(w) + 1e-8)

        pshape(x_t, omegas, v_t, x_t, g_t, lhs, rhs, err2)
        aux['lhs'] = jnp.mean(lhs**2)
        aux['rhs'] = jnp.mean(rhs**2)
        aux['err2'] = jnp.mean(err2)
        aux['t'] = jnp.mean(t)

        if reg_kin > 0:
            kin_loss = jnp.mean(jnp.sum(v_t ** 2, axis=-1))
            aux['kin'] = kin_loss
            final_loss = final_loss + reg_kin * kin_loss

        if reg_traj > 0:
            xp1_pred = x_t + dt*v_t
            err = xp1_pred - xtp1_batch
            pshape(xp1_pred, xtp1_batch, dt)
            traj_loss = jnp.mean(jnp.sum(err ** 2, axis=-1)) / \
                jnp.mean(jnp.sum(xtp1_batch ** 2, axis=-1))
            aux['traj'] = traj_loss
            final_loss = final_loss + reg_traj * traj_loss
        return final_loss, aux

    return loss_fn


def make_gmfm_loss_resample(cfg: Config, apply_fn):

    sigma = cfg.loss.sigma
    reg_kin = cfg.loss.reg_kin
    relative = cfg.loss.relative
    normalize = cfg.loss.normalize
    bs_o = cfg.loss.n_functions

    grad_fn = rff_grad_dot_v
    lap_fn = rff_laplace_phi

    def loss_fn(params, x_t, t, x_tm1, sigma_t, dt, key):
        final_loss = 0.0
        aux = {}
        v_t = apply_fn(params, x_t, t)  # (B, D)
        v_t = rearrange(v_t, 'N ... -> N (...)')
        x_t = rearrange(x_t, 'N ... -> N (...)')

        D = x_t.shape[-1]
        sigma_t = jnp.mean(sigma_t)
        print(sigma_t.shape)

        omegas = make_rff_params_jnp(
            key, bs_o, D, sigma_t)

        mu_t = rff_phi(x_t, omegas)
        mu_tm1 = rff_phi(x_tm1, omegas)
        lhs = (mu_t - mu_tm1) / jnp.mean(dt)

        # E[∇phi · v]
        g_t = grad_fn(x_t, v_t, omegas)  # (2M,)

        # E[Δphi]
        if sigma > 0.0:
            lap_t = lap_fn(x_t, omegas)
            rhs = g_t + 0.5 * (sigma**2) * lap_t
        else:
            rhs = g_t

        resid = (lhs - rhs)**2

        if normalize == 'omega':
            w = jnp.sum(omegas**2, axis=-1)   # (M,)
            w = 1.0 / (w + 1e-8)              # (M,)
            w = jnp.repeat(w, 2)  # (2M,) if sin/cos paired
            resid = resid * jax.lax.stop_gradient(w)
        if normalize == 'grad':
            w = grad_phi_weights(x_t, omegas, eps=1e-8)          # (2M,)
            w = jax.lax.stop_gradient(w)
            resid = resid * w

        pshape(x_t, omegas, v_t, x_t, g_t, lhs, rhs, resid)
        aux['lhs'] = jnp.mean(lhs**2)
        aux['rhs'] = jnp.mean(rhs**2)
        aux['resid'] = jnp.mean(resid)
        aux['t'] = jnp.mean(t)

        if relative:
            num = jnp.mean(resid)
            if normalize is not None:
                den = jnp.mean(lhs**2 * w)
            else:
                den = jnp.mean(lhs**2)
            final_loss += num / (den + 1e-8)
        else:
            final_loss += jnp.mean(resid)

        if reg_kin > 0:
            final_loss = final_loss + reg_kin * \
                jnp.mean(jnp.sum(v_t ** 2, axis=-1))

        return final_loss, aux

    return loss_fn
