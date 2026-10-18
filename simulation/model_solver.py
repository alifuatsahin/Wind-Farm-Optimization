import jax.numpy as jnp

from .utils import div_nu_grad


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


def representative_velocity(U, Uin):
    """Advection speed of the marching step: mean U over "the wake", i.e. the points
    whose deficit exceeds a fifth of the largest on the plane.

    Both the threshold and the average have to be taken over PHYSICALLY ADMISSIBLE
    points. A parabolic march cannot represent reversed flow, so U < 0 is outside the
    model's own validity; but max(Us) is a global reduction, so a single such point
    anywhere sets the threshold for the whole plane. Measured on a two-column farm: one
    corner reached U = -13 m/s, so max(Us) became 21.8 against a free stream of 8, the
    threshold rose to 4.35 and excluded the real wake (Us ~ 3.5), the mask collapsed
    from 1438 points to 9, the representative velocity went to zero and the march
    froze -- no step budget could help. Capping the deficit at Uin (equivalently
    U >= 0) and averaging only over admissible points keeps one bad cell local instead
    of fatal. Healthy fields have U > 0 everywhere, so both are identities there.

    Shared with the diffusive step limit in turbine_physics, which has to know dx to
    bound the pseudo-time dx/U that the diffusion operator actually sees. Keep it one
    definition: a limiter that disagrees with the step it is limiting is no limiter.
    """
    Us = Uin - U
    admissible = U > 0.0
    Us_capped = jnp.minimum(Us, Uin)
    mask = (Us > 0.2 * jnp.max(Us_capped)) & admissible
    n_mask = jnp.maximum(jnp.sum(mask), 1)
    return jnp.sum(jnp.where(mask, U, 0.0)) / n_mask


def velocity_floor(Uin):
    """Lower bound on U wherever it appears in a denominator."""
    return jnp.maximum(0.1 * Uin, 1e-6)


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

    dx = dt * representative_velocity(U, Uin)

    # prediction step
    dUdY, dUdZ = jnp.gradient(U, yloc[:, 0], zloc[0, :])   # dUdY = ∂(U)/∂y, dUdZ = ∂(U)/∂z
    def _diffusion(Ufield):
        """Turbulent diffusion of the DEFICIT, in divergence form.

        Zong & Porte-Agel (2020) Sec 3.1 close the Reynolds stresses as -<uv> = nu_T dU/dy,
        so his Eq (3.5) is genuinely div(nu_T grad U); he then drops grad(nu_T).grad(U)
        legitimately, because he assumes "the eddy viscosity nu_T is ... uniform in the
        cross-stream plane, dependent only on the x-coordinate", and Du's Eq (26) returns
        such a field too. That assumption does not hold here: nu_T is cell-wise and, under
        the marched k_w closure, peaks AT the shear layers where the deficit gradient is
        largest, so the dropped term is first order rather than zero and nu_T*lap(Uw) does
        not conserve the integrated momentum deficit.

        The non-conservative form was carried as an option for one round of measurements and
        then removed: it changed the RANKING of the nu_T closures, not just their scores, so
        keeping it available only invited benchmarking on the wrong equation.

        The STENCIL is utils.div_nu_grad, three-point and flux-form. It used to be
        jnp.gradient applied twice, which is a 2h-lattice operator that damps the
        grid-scale checkerboard mode exactly zero -- see the note on div_nu_grad.
        """
        dev = Ufield - Uin
        return div_nu_grad(NuT, dev, yloc[1, 0] - yloc[0, 0], zloc[0, 1] - zloc[0, 0])

    # assemble numerator
    numer = -V * dUdY - W * dUdZ + _diffusion(U)
    U_floor = velocity_floor(Uin)
    denom_safe = jnp.maximum(U, U_floor)

    dUpdx = numer / denom_safe

    Up = U + dUpdx * dx
    Up = Up.at[:, 0].set(Uin[:, 0])  # ground
    Up = _lateral_bc(Up, Uin)        # lateral far field, non-reflecting

    # correction step
    dUdY, dUdZ = jnp.gradient(Up, yloc[:, 0], zloc[0, :])   # dUdY = ∂(Up)/∂y, dUdZ = ∂(Up)/∂z
    numer = -V * dUdY - W * dUdZ + _diffusion(Up)
    denom_safe = jnp.maximum(Up, U_floor)

    dUcdx = numer / denom_safe
    U = U + 0.5 * (dUpdx + dUcdx) * dx
    U = U.at[:, 0].set(Uin[:, 0])  # ground
    U = _lateral_bc(U, Uin)        # lateral far field, non-reflecting

    return U, data.X + dx
