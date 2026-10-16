import jax.numpy as jnp

def _lateral_bc(U, Uin):
    """Zero-gradient on the DEFICIT at the two lateral edges.

    The only Dirichlet condition this solver applied was U[:, 0] = Uin[:, 0], which on a
    (Ny, Nz) field is every y at the first z -- the GROUND, despite being labelled "far
    stream". The y-edges were left free, so jnp.gradient fell back to one-sided
    differences there, which supports a growing mode. On a +-1.5D grid the edge sits
    inside flow the rotor's own smoothed initial condition keeps well behaved; once the
    grid spans a whole farm the free edge sits in nominally undisturbed flow with room to
    grow, and it did: every cell that diverged in a two-column farm was on a lateral edge.

    Copying the first interior deficit outward is non-reflecting, so a wake that really
    does reach the edge leaves rather than being clamped to zero. The top edge is left
    free on purpose -- the wake genuinely reaches it (11.6% deficit on the wind-tunnel
    case), so a Dirichlet there would truncate real physics.
    """
    U = U.at[0, :].set(Uin[0, :] - (Uin[1, :] - U[1, :]))
    U = U.at[-1, :].set(Uin[-1, :] - (Uin[-2, :] - U[-2, :]))
    return U


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

    # Representative velocity for the marching step, over "the wake": the points whose
    # deficit exceeds a fifth of the largest on the plane.
    #
    # Both the threshold and the average have to be taken over PHYSICALLY ADMISSIBLE
    # points. A parabolic march cannot represent reversed flow, so U < 0 is outside the
    # model's own validity; but max(Us) is a global reduction, so a single such point
    # anywhere sets the threshold for the whole plane. Measured on a two-column farm: one
    # corner reached U = -13 m/s, so max(Us) became 21.8 against a free stream of 8, the
    # threshold rose to 4.35 and excluded the real wake (Us ~ 3.5), the mask collapsed
    # from 1438 points to 9, the representative velocity went to zero and the march
    # froze -- no step budget could help. Capping the deficit at Uin (equivalently
    # U >= 0) and averaging only over admissible points keeps one bad cell local instead
    # of fatal. Healthy fields have U > 0 everywhere, so both are identities there.
    Us = Uin - U
    admissible = U > 0.0
    Us_capped = jnp.minimum(Us, Uin)
    mask = (Us > 0.2 * jnp.max(Us_capped)) & admissible
    n_mask = jnp.maximum(jnp.sum(mask), 1)
    dx = dt * jnp.sum(jnp.where(mask, U, 0.0)) / n_mask

    # prediction step
    dUdY, dUdZ = jnp.gradient(U, yloc[:, 0], zloc[0, :])   # dUdY = ∂(U)/∂y, dUdZ = ∂(U)/∂z
    dUin_dZ = jnp.gradient(Uin, zloc[0, :], axis=1)
    dUin_dY = jnp.gradient(Uin, yloc[:, 0], axis=0)

    Sxy = dUdY
    Sxz = dUdZ

    # derivatives of strains -- each only needs its own axis, not both
    dSxydY = jnp.gradient(Sxy, yloc[:, 0], axis=0)   # = ∂Sxy/∂y
    dSxzdZ = jnp.gradient(Sxz, zloc[0, :], axis=1)   # = ∂Sxz/∂z
    d2Uin_dZ2 = jnp.gradient(dUin_dZ, zloc[0, :], axis=1)
    d2Uin_dY2 = jnp.gradient(dUin_dY, yloc[:, 0], axis=0)
    lap_Uin = d2Uin_dY2 + d2Uin_dZ2

    # assemble numerator
    numer = -V * dUdY - W * dUdZ + NuT * dSxydY + NuT * dSxzdZ - NuT * lap_Uin
    U_floor = jnp.maximum(0.1 * Uin, 1e-6)
    denom_safe = jnp.maximum(U, U_floor)

    dUpdx = numer / denom_safe

    Up = U + dUpdx * dx
    Up = Up.at[:, 0].set(Uin[:, 0])  # ground
    Up = _lateral_bc(Up, Uin)        # lateral far field, non-reflecting

    # correction step
    dUdY, dUdZ = jnp.gradient(Up, yloc[:, 0], zloc[0, :])   # dUdY = ∂(Up)/∂y, dUdZ = ∂(Up)/∂z
    Sxz = dUdZ
    Sxy = dUdY

    dSxydY = jnp.gradient(Sxy, yloc[:, 0], axis=0)   # = ∂Sxy/∂y
    dSxzdZ = jnp.gradient(Sxz, zloc[0, :], axis=1)   # = ∂Sxz/∂z
    numer = -V * dUdY - W * dUdZ + NuT * dSxydY + NuT * dSxzdZ - NuT * lap_Uin
    denom_safe = jnp.maximum(Up, U_floor)

    dUcdx = numer / denom_safe
    U = U + 0.5 * (dUpdx + dUcdx) * dx
    U = U.at[:, 0].set(Uin[:, 0])  # ground
    U = _lateral_bc(U, Uin)        # lateral far field, non-reflecting

    return U, data.X + dx
