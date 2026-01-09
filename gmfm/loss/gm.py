import jax.numpy as jnp
from einops import rearrange

from gmfm.config.config import Config
from gmfm.loss.rff import (
    rff_grad_dot_v,
    rff_grad_dot_v_chunked,
    rff_laplace_phi,
    rff_laplace_phi_chunked,
)
from gmfm.utils.tools import pshape


def make_gmfm_loss(cfg: Config, apply_fn):

    sigma = cfg.loss.sigma
    reg_kin = cfg.loss.reg_kin
    relative = cfg.loss.relative

    chunk = False
    if chunk:
        def grad_fn(x, v, om): return rff_grad_dot_v_chunked(
            x, v, om, chunk=100)
        def lap_fn(x,    om): return rff_laplace_phi_chunked(
            x,    om, chunk=100)
    else:
        grad_fn = rff_grad_dot_v
        lap_fn = rff_laplace_phi

    def loss_fn(params, x_t, omegas, lhs, t, dt, key):
        final_loss = 0.0
        aux = {}

        v_t = apply_fn(params, x_t, t)  # (B, D)
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

        resid = lhs - rhs

        pshape(x_t, omegas, v_t, x_t, g_t, lhs, rhs, resid)
        aux['lhs'] = jnp.mean(lhs**2)
        aux['rhs'] = jnp.mean(rhs**2)
        aux['resid'] = jnp.mean(resid**2)
        aux['t'] = jnp.mean(t)

        if relative:
            num = jnp.mean(resid**2)
            den = jnp.mean(lhs**2)
            final_loss += num / (den + 1e-8)
        else:
            final_loss += jnp.mean(resid**2)

        if reg_kin > 0:
            final_loss = final_loss + reg_kin * \
                jnp.mean(jnp.sum(v_t ** 2, axis=-1))

        return final_loss, aux

    return loss_fn
