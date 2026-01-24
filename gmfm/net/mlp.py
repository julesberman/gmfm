

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange


import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange


class DNN(nn.Module):
    width: int
    depth: int
    out_features: int
    activate_last: bool = False
    residual: bool = False
    n_classes: int = 10
    use_bias: bool = True
    n_harmonics: int = 0

    @nn.compact
    def __call__(self, x, time, mu):

        x_shape = x.shape
        if x.ndim == 1:
            x = x[None]
        if time is not None and time.ndim == 1:
            time = time[None]

        x = rearrange(x, 'B ... -> B (...)')  # (B, D_flat)

        # NEW: periodic Fourier feature embedding on [-1,1] (period 2)
        # Uses harmonics m=1..n_harmonics:
        #   sin(pi*m*x), cos(pi*m*x)
        # This guarantees invariance under x -> x + 2 e_i for each coordinate.
        if self.n_harmonics > 0:
            H = int(self.n_harmonics)
            m = jnp.arange(1, H + 1, dtype=x.dtype)              # (H,)
            # (B, D_flat, H)
            xm = x[..., None] * m[None, None, :]
            # (B, D_flat, H)
            ang = jnp.pi * xm

            sinx = jnp.sin(ang)
            cosx = jnp.cos(ang)

            # (B, D_flat, 2H) -> (B, D_flat*2H)
            x = jnp.concatenate([sinx, cosx], axis=-1).reshape(x.shape[0], -1)

        A = nn.gelu

        if time is not None:
            time = MLP(self.width, depth=2,
                       out_features=self.out_features, activate_last=True)(time)

        if mu is not None:
            mu = MLP(self.width, depth=2,
                     out_features=self.out_features, activate_last=True)(mu)

        temb = None
        if time is not None and mu is not None:
            temb = jnp.concatenate([time, mu], axis=-1)
        if time is None and mu is not None:
            temb = mu
        if time is not None and mu is None:
            temb = time

        last_x = None
        for _ in range(self.depth):
            D = nn.Dense(self.width, use_bias=self.use_bias)

            last_x = x
            x = D(x)
            x = A(x)

            if temb is not None:
                bias = MLP(width=self.width, depth=2)(temb)
                bias = bias.reshape(-1, self.width)
                x = x + bias

            if self.residual and last_x is not None:
                if x.shape == last_x.shape:
                    x = x + last_x

        # Always single head
        x = nn.Dense(self.out_features, use_bias=self.use_bias)(x)

        x = x.reshape((*x_shape[:-1], self.out_features))
        return x


class MLP(nn.Module):
    width: int
    depth: int
    out_features: int | None = None
    use_bias: bool = True
    activate_last: bool = False

    @nn.compact
    def __call__(self, x):
        A = nn.gelu
        out_features = self.out_features
        if out_features is None:
            out_features = self.width

        for _ in range(self.depth - 1):
            D = nn.Dense(
                self.width,
                use_bias=self.use_bias,
            )
            x = D(x)
            x = A(x)

        D = nn.Dense(
            out_features,
            use_bias=self.use_bias,
        )
        x = D(x)

        if self.activate_last:
            x = A(x)

        return x
