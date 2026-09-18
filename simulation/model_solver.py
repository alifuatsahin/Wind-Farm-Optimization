import jax.numpy as jnp

def advance_wake_field(data, dt, NuT, config, WV):
    """Update the streamwise velocity according to transportation-diffusion equations."""
    Uin = config.Uin
    U = data.U
    V = data.V
    W = data.W
    yloc = data.yloc
    zloc = data.zloc

    V_veer = U * jnp.sin(jnp.deg2rad(WV * (zloc - config.Zhub)))
    V = V + V_veer # total transverse velocity

    Us = Uin - U
    mask = Us > 0.2 * jnp.max(Us)
    dx = dt * jnp.sum(jnp.where(mask, U, 0.0)) / jnp.sum(mask)

    # prediction step
    dUdY, dUdZ = jnp.gradient(U, yloc[:, 0], zloc[0, :])   # dUdY = ∂(U)/∂y, dUdZ = ∂(U)/∂z
    # Gradient of Uin (Background Shear) -- only the z-direction is used downstream
    dUin_dZ = jnp.gradient(Uin, zloc[0, :], axis=1)

    Sxy = dUdY
    Sxz = dUdZ

    # derivatives of strains -- each only needs its own axis, not both
    dSxydY = jnp.gradient(Sxy, yloc[:, 0], axis=0)   # = ∂Sxy/∂y
    dSxzdZ = jnp.gradient(Sxz, zloc[0, :], axis=1)   # = ∂Sxz/∂z
    d2Uin_dZ2 = jnp.gradient(dUin_dZ, zloc[0, :], axis=1)

    # assemble numerator
    numer = -V * dUdY - W * dUdZ + NuT * dSxydY + NuT * dSxzdZ - NuT * d2Uin_dZ2
    denom_safe = jnp.maximum(U, 1e-6)  # prevent division by zero

    dUpdx = numer / denom_safe
    Up = U + dUpdx * dx
    Up = Up.at[:, 0].set(Uin[:, 0]) # far stream dirichlet BC

    # correction step
    dUdY, dUdZ = jnp.gradient(Up, yloc[:, 0], zloc[0, :])   # dUdY = ∂(Up)/∂y, dUdZ = ∂(Up)/∂z
    Sxz = dUdZ
    Sxy = dUdY

    dSxydY = jnp.gradient(Sxy, yloc[:, 0], axis=0)   # = ∂Sxy/∂y
    dSxzdZ = jnp.gradient(Sxz, zloc[0, :], axis=1)   # = ∂Sxz/∂z
    numer = -V * dUdY - W * dUdZ + NuT * dSxydY + NuT * dSxzdZ - NuT * d2Uin_dZ2
    denom_safe = jnp.maximum(Up, 1e-6)  # prevent division by zero

    dUcdx = numer / denom_safe
    U = U + 0.5 * (dUpdx + dUcdx) * dx
    U = U.at[:, 0].set(Uin[:, 0]) # far stream dirichlet BC

    return U, data.X + dx
