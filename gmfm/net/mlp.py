

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange


class DNN(nn.Module):
    width: int
    depth: int
    out_features: int
    activate_last: bool = False
    residual: bool = False
    n_classes: int = 10
    use_bias: bool = True
    heads: int = 1

    @nn.compact
    def __call__(self,  x, time, mu):

        x_shape = x.shape
        x = rearrange(x, 'B ... -> B (...)')

        A = nn.gelu
        if time is not None:
            time = MLP(self.width, depth=2,
                       out_features=self.out_features, activate_last=True)(time)

        if mu is not None:
            mu = MLP(self.width, depth=2,
                     out_features=self.out_features, activate_last=True)(mu)

        temb = None
        if time is not None and mu is not None:
            temb = jnp.concatenate(
                [time, mu], axis=-1)
        if time is None and mu is not None:
            temb = mu
        if time is not None and mu is None:
            temb = time
        # if temb is not None:
        #     temb = MLP(
        #         width=self.width,
        #         depth=1,
        #         use_bias=self.use_bias,
        #     )(temb)

        last_x = None
        for _ in range(self.depth):
            D = nn.Dense(
                self.width,
                use_bias=self.use_bias,
            )

            last_x = x
            x = D(x)
            x = A(x)

            if temb is not None:
                bias = MLP(
                    width=self.width,
                    depth=2,
                )(temb)

                bias = bias.reshape(-1,  self.width)
                x = x + bias

            if self.residual and last_x is not None:
                if x.shape == last_x.shape:
                    x = x + last_x

        if self.heads == 1:
            x = nn.Dense(
                self.out_features,
                use_bias=self.use_bias,
            )(x)
        else:
            xs = []
            for _ in range(self.heads):
                x_c = MLP(self.width, depth=3,
                          out_features=self.out_features)(x)
                xs.append(x_c)
            x = jnp.asarray(xs)

        x = x.reshape(x_shape)

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
