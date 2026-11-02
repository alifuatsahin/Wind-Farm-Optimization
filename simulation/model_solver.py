import jax.numpy as jnp

from .utils import div_nu_grad, velocity_floor


def _lateral_bc(U, Uin):
    """Zero-gradient on the DEFICIT at the two lateral edges."""
    U = U.at[0, :].set(Uin[0, :] - (Uin[1, :] - U[1, :]))
    U = U.at[-1, :].set(Uin[-1, :] - (Uin[-2, :] - U[-2, :]))
    return U


def representative_velocity(U, Uin):
    """Advection speed of the marching step: mean U over "the wake", i.e. the points whose deficit..."""
    Us = Uin - U
    admissible = U > 0.0
    Us_capped = jnp.minimum(Us, Uin)
    mask = (Us > 0.2 * jnp.max(Us_capped)) & admissible
    n_mask = jnp.maximum(jnp.sum(mask), 1)
    return jnp.sum(jnp.where(mask, U, 0.0)) / n_mask


def advance_wake_field(data, dt, NuT, config, WV):
    """Update the streamwise velocity according to transportation-diffusion equations."""
    Uin = config.Uin
    U = data.U
    V = data.V
    W = data.W
    yloc = data.yloc
    zloc = data.zloc

    V_veer = U * jnp.sin(jnp.deg2rad(WV * (zloc - config.Zhub)))
    V = V + V_veer

    dx = dt * representative_velocity(U, Uin)

    dUdY, dUdZ = jnp.gradient(U, yloc[:, 0], zloc[0, :])
    def _diffusion(Ufield):
        """Turbulent diffusion of the DEFICIT, in divergence form."""
        dev = Ufield - Uin
        return div_nu_grad(NuT, dev, yloc[1, 0] - yloc[0, 0], zloc[0, 1] - zloc[0, 0])

    numer = -V * dUdY - W * dUdZ + _diffusion(U)
    U_floor = velocity_floor(Uin)
    denom_safe = jnp.maximum(U, U_floor)

    dUpdx = numer / denom_safe

    Up = U + dUpdx * dx
    Up = Up.at[:, 0].set(Uin[:, 0])
    Up = _lateral_bc(Up, Uin)

    dUdY, dUdZ = jnp.gradient(Up, yloc[:, 0], zloc[0, :])
    numer = -V * dUdY - W * dUdZ + _diffusion(Up)
    denom_safe = jnp.maximum(Up, U_floor)

    dUcdx = numer / denom_safe
    U = U + 0.5 * (dUpdx + dUcdx) * dx
    U = U.at[:, 0].set(Uin[:, 0])
    U = _lateral_bc(U, Uin)

    return U, data.X + dx, V, W
