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
from gmfm.utils.tools import pshape, time_total_derivative, hutch_frob_jac, hutch_antisym_frob, hutch_div2


def make_gmfm_loss(cfg: Config, apply_fn):

    sigma = cfg.loss.sigma
    reg_amt = cfg.loss.reg_amt
    reg_type = cfg.loss.reg_type

    normalize = cfg.loss.normalize
    has_mu = cfg.data.has_mu

    grad_fn = rff_grad_dot_v
    lap_fn = rff_laplace_phi

    dv_dt = time_total_derivative(lambda p, x, t, mu: apply_fn(
        p, x, t, mu), argnum_x=1, argnum_t=2, stopgrad_dx=True)

    def loss_fn(params, x_t, t, omegas, lhs, mu, xtp1_batch, dt, key):
        final_loss = 0.0
        aux = {}
        # Keep raw tensors around
        x_t_raw = x_t                          # (B,W,H,C) or (B,D)
        B = x_t_raw.shape[0]
        # () for (B,D) case? actually (D,) then
        x_shape = x_t_raw.shape[1:]
        x_t_flat = x_t_raw.reshape(B, -1)      # (B, Dflat)

        if not has_mu:
            mu = None

        v_t = apply_fn(params, x_t, t, mu)

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
            final_loss = jnp.sum(w * err2) / (jnp.sum(w * lhs**2) + 1e-8)
        elif normalize == 'grad':
            w = grad_phi_weights(x_t, omegas, eps=1e-8)        # (2M,)
            w = w / (jnp.mean(w) + 1e-8)                       # optional
            w = jnp.clip(w, 0.0, 1e3)                          # optional
            w = jax.lax.stop_gradient(w)
            final_loss = jnp.sum(w * err2) / (jnp.sum(w * lhs**2) + 1e-8)
        elif normalize == 'lhs':
            err2 = (lhs - rhs)**2
            den = jnp.mean(lhs**2)
            den = jax.lax.stop_gradient(den)
            final_loss = jnp.mean(err2) / (den + 1e-8)
        elif normalize == 'rhs':
            err2 = (lhs - rhs)**2
            den = jnp.mean(rhs**2)
            den = jax.lax.stop_gradient(den)
            final_loss = jnp.mean(err2) / (den + 1e-8)
        elif normalize == 'sym':
            err2 = (lhs - rhs)**2
            den = jnp.mean(lhs**2) + jnp.mean(rhs**2)
            den = jax.lax.stop_gradient(den)
            final_loss = jnp.mean(err2) / (den + 1e-8)
        else:
            err2 = (lhs - rhs)**2
            final_loss = jnp.mean(err2)

        pshape(x_t, omegas, v_t, x_t, g_t, lhs, rhs, err2)
        aux['lhs'] = jnp.mean(lhs**2)
        aux['rhs'] = jnp.mean(rhs**2)
        aux['err2'] = jnp.mean(err2)
        aux['t'] = jnp.mean(t)
        aux['dt'] = jnp.mean(dt)

        if reg_amt > 0:
            if reg_type == 'kin':
                kin_loss = jnp.mean(jnp.sum(v_t ** 2, axis=-1))
                aux['kin'] = kin_loss
                final_loss = final_loss + reg_amt * kin_loss

            if reg_type == 'curl':
                def v_of_xt(xt_, t_):
                    return apply_fn(params, xt_, t_, mu)  # (B,D)

                anti_reg = hutch_antisym_frob(v_of_xt, argnum=0)

                anti_per_sample = anti_reg(x_t, t, key=key)   # (B,)
                curl_loss = jnp.mean(anti_per_sample)         # scalar
                aux['curl'] = curl_loss
                final_loss = final_loss + reg_amt * curl_loss

            if reg_type == 'div':

                def v_of_xt(xt_flat, t_):
                    # xt_flat: (B, Dflat)
                    B_ = xt_flat.shape[0]

                    # Unflatten for the network if needed
                    xt_in = xt_flat.reshape(
                        (B_,) + x_shape)   # (B,W,H,C) or (B,D)

                    # should be same shape as xt_in
                    v_out = apply_fn(params, xt_in, t_, mu)

                    # Flatten output back to (B, Dflat) so hutch_div2's axis=-1 reduction yields (B,)
                    return v_out.reshape(B_, -1)

                div2_reg = hutch_div2(v_of_xt, argnum=0, unbiased=False)

                div2_per_sample = div2_reg(x_t_flat, t, key=key)   # (B,)
                div_pen = jnp.mean(div2_per_sample)

                aux['div'] = div_pen
                final_loss = final_loss + reg_amt * div_pen

            if reg_type == 'traj':
                key_time, _ = jax.random.split(key, 2)
                dv = dv_dt(params, x_t, t, mu, key=key_time, material=True)
                traj_loss = jnp.mean(jnp.sum(dv * dv, axis=-1))
                aux['traj'] = traj_loss
                final_loss = final_loss + reg_amt * traj_loss

            if reg_type == 'grad':
                def v_of_xt(xt_, t_):
                    return apply_fn(params, xt_, t_, mu)  # (B,D)
                frob_reg = hutch_frob_jac(v_of_xt, argnum=0)
                frob_per_sample = frob_reg(x_t, t, key=key)
                smooth_pen = jnp.mean(frob_per_sample)  # scalar ≈ E[||J||_F^2]
                aux['grad'] = smooth_pen
                final_loss = final_loss + reg_amt * smooth_pen

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
