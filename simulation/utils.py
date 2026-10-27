from scipy.ndimage import uniform_filter, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patheffects as patheffects
import numpy as np
import jax.numpy as jnp
import os

from .field_ops import interpolate_local_velocity_field
from .viz_style import SURFACE, INK_PRIMARY, INK_SECONDARY, style_figure, style_axes, style_colorbar

DU_KV1, DU_KV0, DU_XV1 = 0.05, 0.001, 0.5
IU_TO_TI = 1.28   # Du et al. Sec 2.3: Iu ~ 1.28 TI under neutral conditions


def du_nu_t(TI, age_D, U0, D):
    """Du Eq (26): nu_t/(U0 D) = (kv1 TI - kv0) * min(age/D, xv1/TI)."""
    return (jnp.maximum(DU_KV1 * TI - DU_KV0, 0.0)
            * jnp.minimum(age_D, DU_XV1 / TI) * U0 * D)


def ambient_tke(I_amb, U0):
    """k_B from TI = sqrt(2k/3)/U0, with TI the ambient value Iu/1.28."""
    return 1.5 * ((I_amb / IU_TO_TI) * U0) ** 2


# ---------------------------------------------------------------------------------------
# COMPACT cross-plane diffusion operator.
#
# jnp.gradient applied twice is NOT a usable second derivative here. Central-differencing
# twice gives (f[i+2] - 2 f[i] + f[i-2]) / (4h^2) -- a stencil on the 2h lattice, which
# decouples the odd and even sublattices. The grid-scale checkerboard mode is then damped
# EXACTLY zero, so any checkerboard noise is free to grow. Measured on an alternating
# field: the double-gradient operator returns 0 where a three-point operator returns
# -4/h^2, while the two agree to ~1% on smooth fields.
#
# With Du's nu_T, which is small and smooth, the mode never gets excited and the error is
# invisible -- which is why this stood for so long. It is not invisible for a larger or
# shear-peaked eddy viscosity. model_solver._lateral_bc already exists because the same
# growing mode appeared at the lateral edges, where jnp.gradient falls back to one-sided
# differences; this is the interior form of the same defect.
#
# The boundary ring carries only its TANGENTIAL term. The y-flux divergence is written to
# rows 1:-1 of every column and the z-flux divergence to columns 1:-1 of every row, so the
# top row still diffuses laterally and the lateral columns still diffuse vertically -- what
# is zero at each edge is the NORMAL flux. Three of the four edges are overwritten after
# every step anyway (the ground by a Dirichlet condition, the two lateral edges by
# _lateral_bc), so only the top keeps a condition of its own, and that condition is now zero
# vertical flux -- a perfectly reflecting lid, where the old one-sided jnp.gradient leaked.
# Measured on the 8-turbine row, raising max_Z from 2D to 4D moves R2 0.9195 -> 0.9188 and
# T8's Uhub/U0 0.6309 -> 0.6238, so the 2D lid retains about 0.007 of deficit at the last
# rotor. One-signed and growing down the row, hence it flatters the deep-array result
# slightly; small enough to leave, but it is a physical choice now rather than a by-product
# of the stencil. (The earlier max_Z 2 -> 4 check, which moved R2 by 0.001, predates this
# operator and was made with a leaking top.)
#
# This is also what licenses the reduction in turbine_physics' diffusive step limit: the
# ring cannot go unstable in the normal direction, and in the tangential one it is bounded
# by the looser one-dimensional limit at the LARGEST U on the plane, so that limit takes its
# maximum over the interior only.
# ---------------------------------------------------------------------------------------

def div_nu_grad(nu, f, dy, dz):
    """div(nu grad f) on the three-point stencil, in conservative flux form.

    nu is evaluated at the cell faces as the arithmetic mean of its neighbours, so the
    integrated flux telescopes and the scheme conserves exactly -- the property the
    divergence form was adopted for (see model_solver.advance_wake_field._diffusion).
    """
    qy = 0.5 * (nu[:-1, :] + nu[1:, :]) * (f[1:, :] - f[:-1, :]) / dy
    out = jnp.zeros_like(f).at[1:-1, :].add((qy[1:, :] - qy[:-1, :]) / dy)
    qz = 0.5 * (nu[:, :-1] + nu[:, 1:]) * (f[:, 1:] - f[:, :-1]) / dz
    return out.at[:, 1:-1].add((qz[:, 1:] - qz[:, :-1]) / dz)


def velocity_floor(Uin):
    """Lower bound on U wherever it appears in a denominator.

    Lives here rather than in model_solver because the momentum march and the k_w budget
    both divide by the local velocity and must floor it identically; utils is the module
    both can import.
    """
    return jnp.maximum(0.1 * Uin, 1e-6)


def du_eps_coef(TI, D):
    """Du's dissipation coefficient, from his C_eps and l_m: eps = C_eps k_w^(3/2) / l_m.

    He gives C_eps = (0.4 TI + 0.010) x/D and l_m/D = (0.2 TI + 0.015) x/D (Eq 27 with the
    Fig 11 slopes), so the streamwise factor CANCELS and the coefficient is
    AGE-INDEPENDENT -- one fewer place where "age since which rotor" has to be decided.

    This replaces Du's own Psi = c l_m^2 / C_eps, which exists only to linearise the
    budget; see the dissipation note in advance_kw. Dropping it also drops c = 0.67, the
    one constant of his whose provenance was awkward here: he defines it by dividing by
    sqrt(K_w,max), the MAXIMUM azimuthally-averaged k_w, and states it is really c(x, r)
    collapsed for tractability -- so using it cell-wise was an assumption we no longer
    need to make. The two published slopes are unchanged.
    """
    return (0.4 * TI + 0.010) / ((0.2 * TI + 0.015) * D)


def advance_kw(kw, U, Uin, yloc, zloc, dx, nu_t, eps_coef, V, W):
    """One march step of Du et al. (2025) Eq (13), the wake-added TKE budget:

        u dk_w/dx = div_perp(nu_t grad_perp k_w) - C_eps k_w^(3/2)/l_m + nu_t |grad_perp U|^2
                    - V dk_w/dy - W dk_w/dz

    i.e. cross-plane diffusion, dissipation, and shear production. Du's Eq (14) is the
    analytic solution of this same equation; marching it instead avoids that solution's
    modified-Bessel approximation, its unreconciled Psi vs psi, and the undefined delta in
    its integration limit -- and because the cross-plane here is genuinely two-dimensional,
    the operator is just the 2-D Laplacian, with no axis singularity and NO axisymmetry
    assumption. Du's own formulation is axisymmetric in <k_w>_theta and therefore cannot
    represent the unequal lateral peaks the measurements show; this can.

    Du writes the first term as `nu_t grad^2 k_w`, and for him the two forms are the same
    equation: his Eq (26) returns an eddy viscosity that is a function of x and the rotor's
    inflow TI alone, so grad_perp(nu_t) = 0 identically. Here nu_t is Eq (26) evaluated on
    the incident-TI FIELD, which varies across the plane, so the forms differ by
    grad_perp(nu_t).grad_perp(k_w) -- first order, not zero. The term being modelled is the
    divergence of the turbulent flux of k, so the divergence form is the faithful one; this
    is the same argument model_solver.advance_wake_field._diffusion makes for the momentum
    equation, which it reached first. Carrying it over here moves everything the same way
    and nothing the other: 8-turbine R2 .9195 -> .9235 and bias -.0194 -> -.0185, LES
    aligned .9263 -> .9266, full-wake C/A/B .9343/.8878/.8989 -> .9356/.8887/.8998, and the
    TI the budget is actually scored on, R2 +.759/+.703/+.680 -> +.782/+.714/+.692 with the
    case-C bias -.0076 -> -.0066. Deflection and runtime are unchanged. Deep in the array it
    goes the right way at every rotor (T4 Uhub/U0 .6609 -> .6708, T8 .6309 -> .6345 against
    a measured ~.643), recovering about a fifth of the k_w under-prediction that the wake-TI
    rescaling exposed -- not a fix for the deep-array limit, which is a mean-flow one, but
    free.

    Replaces the Crespo & Hernandez correlation entirely. Two things go with it: there is
    no superposition rule, because k_w is transported and each rotor simply adds production
    to a field already being carried; and the correlation's lateral structure -- which was
    none at all, one number per plane -- becomes the structure of the solved shear.
    """
    # Production comes from the gradient of the WAKE DEFICIT, not of the total velocity.
    # k_w = k - k^B is the ADDED TKE, so the ambient shear that sustains k^B must not
    # appear here. Using grad(U) instead injects the ABL's own dUin/dz across the whole
    # domain and manufactures k_w outside the wake: measured as a uniform +0.019 bias in
    # I_u everywhere, including the undisturbed far field where k_w must be exactly zero.
    Uw = Uin - U
    dWdY, dWdZ = jnp.gradient(Uw, yloc[:, 0], zloc[0, :])
    prod = nu_t * (dWdY ** 2 + dWdZ ** 2)
    dif = div_nu_grad(nu_t, kw, yloc[1, 0] - yloc[0, 0], zloc[0, 1] - zloc[0, 0])
    # CROSS-PLANE ADVECTION by (V, W), which the momentum equation has carried all along
    # as -V dU/dy - W dU/dz and this budget did not. Without it a yawed wake steers its
    # momentum deficit and leaves its turbulence behind. Du's Eq (13) is axisymmetric and
    # unyawed, so there was no cross-flow in it to carry; the omission is ours, from
    # extending his budget to yawed wakes, and it is the same consistency argument the
    # advection speed below rests on -- k_w is a transported scalar in the same velocity
    # field as U, so it must be transported by all of it. Same jnp.gradient stencil the
    # momentum advection uses, and the same V, that march returns, veer included.
    dkdY, dkdZ = jnp.gradient(kw, yloc[:, 0], zloc[0, :])
    # DISSIPATION, in Du's UNLINEARISED form: C_eps k_w^(3/2) / l_m.
    #
    # He states that physical form, then substitutes (nu_t/Psi) k_w with Psi = c l_m^2/C_eps,
    # noting the two are the same "since nu_t can be written in the form c <k_w>^(1/2) l_m".
    # They are identical only under that Prandtl-Kolmogorov relation, which diverges ~40%
    # from his own Eq (26) once the TI leaves his 0.041-0.071 calibration range -- so the
    # linearised form carries an inconsistency at a waked rotor, where TI ~ 0.13.
    #
    # He linearises for ONE reason: "Eq. (13) is a second-order non-homogeneous LINEAR
    # differential equation", which is what admits his Green's-function solution Eq (14).
    # We march the field, so we do not need linearity and can keep the physical k^(3/2).
    #
    # The equilibrium exponent is what changes: P = eps gives k_eq = Psi S^2 under the
    # linearised form (k ~ S^2) but k_eq ~ (nu_t l_m S^2 / C_eps)^(2/3) here (k ~ S^(4/3)).
    # Down an 8-turbine row the deficit gradient collapses 6x, which costs 36x of
    # equilibrium TKE under S^2 and 11x under S^(4/3) -- so the measured TI rise down the
    # row stops being unreachable. Measured: the model's per-bay effective nu_T stops
    # DECLINING (bays 3-7 were .0071 .0072 .0071 .0069 .0068) and instead rises then
    # flattens (.0075 .0080 .0080 .0077 .0076) against a measurement that rises to ~.0141.
    # The trend is corrected; the magnitude is still about half, which is Eq (26)'s
    # out-of-range problem and not this term's.
    #
    # Full-wake TI R2 +.028/+.006/+.032 with the bias to within .002 of zero on all three;
    # 8-turbine R2 +.008 and power RMSE -17%, with T4-T8 all +.04 to +.06 toward unity.
    rhs = dif - eps_coef * jnp.maximum(kw, 0.0) ** 1.5 + prod - V * dkdY - W * dkdZ

    # ADVECTION SPEED: the LOCAL U, floored exactly as the momentum march floors it.
    #
    # This used to read dx/U0, advancing k_w at the scalar nominal hub speed. The
    # transport equation is u dk/dx, with the local u, and advance_wake_field does
    # precisely that for momentum -- numer / max(U, velocity_floor(Uin)). Du writes
    # U0 dk/dx in his Eq (13), but that is a far-wake simplification inside an ANALYTIC
    # solution; in a marched PDE sitting beside a momentum equation that uses the local U
    # it is simply inconsistent, and it under-evolves k_w by U0/U = 1.4-2.2x exactly
    # where the production lives. Measured as the modelled TI peak over the measured one,
    # at 2D/3D/7D/8D: full-wake case C 0.794/0.794/0.793/0.748 -> 0.941/0.933/0.857/0.772,
    # case A 0.654/0.741/0.726/0.806 -> 0.759/0.838/0.772/0.835. All eight improve, and
    # the second-moment widths move with them rather than against (case C 0.95-1.00 of
    # the measured width, against 0.97-1.08 before), so this is a level correction and
    # not a narrowing that fakes one.
    kw = jnp.maximum(kw + (dx / jnp.maximum(U, velocity_floor(Uin))) * rhs, 0.0)

    # ...and a BOUNDARY CONDITION, which this equation never had.
    #
    # div_nu_grad adds the y-flux on [1:-1, :] -- every z, the GROUND ROW INCLUDED -- and
    # the z-flux on [:, 1:-1]; only the four corners are diffusion-free. The ground row
    # therefore diffuses in y, and it is where the log profile sends U to zero, so once
    # k_w divides by the local U that row's explicit diffusion number blows past the
    # stability limit. Measured in bay 1 of full-wake case C: the limiter holds the
    # interior at exactly DIFFUSIVE_CFL = 0.900 while the ground row runs at 1.155, with
    # U = 0.0145 m/s there against 5.44 m/s at the top -- and the top row, at 0.125, is
    # safe. Momentum never had the problem because advance_wake_field resets the ground
    # to Uin every step, so whatever the over-limit diffusion does there is discarded.
    # k_w had no condition at all, and that omission -- not the advection speed -- is
    # what made the local-U form diverge to 1e17 and stall the LES array march.
    #
    # The condition is ZERO FLUX, not zero value. The ground is impermeable, so the
    # turbulent flux of k through it vanishes; that is Neumann. Momentum can afford
    # Dirichlet there only because Zong's formulation PRESCRIBES U_in(z), so U has a
    # known boundary value; k_w has none. The distinction is not cosmetic -- Dirichlet
    # k_w = 0 drains wake-added TKE into the ground, and it was built and measured:
    # 8-turbine R2 .9306 -> .9239, against .9320 for Neumann, with the per-rotor
    # model/measured power ratio falling .734 -> .701, .657 -> .615 and .601 -> .553 at
    # T5-T8 where Neumann raises every one of them. The lateral edges take _lateral_bc's
    # non-reflecting zero gradient, which is the same statement sideways; the top stays
    # free, as it does for momentum. With the evolved sets of the two equations now
    # identical, the interior-only max in turbine_physics' step limiter is the right
    # domain for both.
    #
    # Bounding that limiter over the full field instead would be the honest alternative,
    # but it is priced: max(nu_T/U) over the whole plane is 0.792 against 0.0047 over the
    # interior, so dt would fall 170x and the first station would not march at all --
    # exactly the failure the limiter's own comment already records.
    kw = kw.at[:, 0].set(kw[:, 1])
    kw = kw.at[0, :].set(kw[1, :])
    return kw.at[-1, :].set(kw[-2, :])


def advect_incident_ti(ti, U, Uin, yloc, zloc, dx, V, W, nu_t=None):
    """Carry the incident-TI field with the flow:

        u dTI/dx = -V dTI/dy - W dTI/dz + div(nu_t grad TI)

    TI_in is the turbulence the shear layer at a cell was BORN into, read from the k_w the
    station was handed and then held fixed in value for the whole segment. Holding the
    VALUE fixed is what breaks the nu_t <-> k_w loop: nothing inside the march may feed
    back on it. Holding its POSITION fixed was never intended and is not implied -- it is
    a label on fluid, and a yawed wake carries its fluid sideways while the label stayed
    behind. Measured on full-wake case A: the wake centre steers 0.10-0.20 D out of the
    frozen TI peak within one 5 D segment, which costs 2.9-5.8% of Du Eq (26)'s prefactor
    at T2 and T3, always NEGATIVE -- a steered wake is systematically under-mixed. Bay 1
    is exactly 0.0%, since its TI_in is uniform ambient and a shift of a constant is a
    constant.

    AND DIFFUSED, which reverses the reasoning this function shipped with. The original
    argument -- that diffusing the label would mix TI between parcels of different birth
    -- ignores that A WAKE WIDENS BY TURBULENT DIFFUSION, NOT BY ADVECTION. Advection
    alone therefore cannot let the label follow the wake it belongs to, and the measured
    consequence was severe: the label is blended at each station by Uw_new, the deficit
    THAT rotor carves, so outside the carved disc a cell keeps its previous value, which
    recurses back to the first rotor and to ambient. The label fell to ambient at the
    DISC edge while the wake had long outgrown it. On a 6-turbine row at 5 D, hub height,
    at the last rotor (ambient TI 0.0602):

        y/D           0.0     0.3     0.5     0.6     0.8     1.4
        incident TI  .1083   .1144   .1041   .0650   .0616   .0615
        deficit/U0    .482    .469    .371    .277    .201    .097

    A factor 1.6 across ONE grid interval at y/D = 0.5, and beyond it a 0.20 U0 deficit
    sitting in flow labelled ambient. Over that annulus -- which carries the entrainment
    and most of the shear production -- Eq (26) was supplying 0.00208 against 0.00420,
    half the eddy viscosity it should. The gap widens down the row as the wake outgrows
    the disc that resets the label (the deficit at y/D = 0.8 goes .048 at T2 to .201 at
    T6 while the TI there stays pinned). It was also a 1.6x coefficient jump across one
    cell inside div_nu_grad, so the grad(nu).grad(k) term the divergence form exists to
    capture was dominated by an artefact.

    NO CIRCULARITY IS REINTRODUCED. The freeze exists to stop nu_t <-> k_w, and the label
    is still never recomputed from the evolving k_w. Diffusing it makes its own spreading
    rate depend on nu_t, which is a nonlinearity, not that loop.

    Same stencil, same local advection speed and the same (V, W) and nu_t as advance_kw,
    for the same reason -- the label rides in the field the momentum march returns, and
    spreads with the same turbulence that spreads the deficit it labels.
    """
    dtiY, dtiZ = jnp.gradient(ti, yloc[:, 0], zloc[0, :])
    rhs = -V * dtiY - W * dtiZ
    if nu_t is not None:
        rhs = rhs + div_nu_grad(nu_t, ti, yloc[1, 0] - yloc[0, 0], zloc[0, 1] - zloc[0, 0])
    ti = ti + (dx / jnp.maximum(U, velocity_floor(Uin))) * rhs
    # Same zero-gradient edges as k_w: a label cannot be created at a boundary.
    ti = ti.at[:, 0].set(ti[:, 1])
    ti = ti.at[0, :].set(ti[1, :])
    return jnp.maximum(ti.at[-1, :].set(ti[-2, :]), 0.0)


def smooth_2d(U, kernel_size=3, method='gaussian'):
    """
    2D smoothing with options:
      - method='gaussian' : Gaussian blur (smooth, continuous look). sigma ~ kernel_size/2.
      - method='uniform'  : equivalent box filter (fast).
      - method='nanmean'  : box filter that ignores NaNs (safe for masked data).

    Returns an array same-shape as U. Keeps NaNs where no valid neighbors exist (for nanmean).
    """
    U = np.asarray(U, dtype=float)
    if method == 'uniform':
        return uniform_filter(U, size=kernel_size, mode='reflect')
    elif method == 'gaussian':
        sigma = max(0.5, kernel_size / 2.0)
        return gaussian_filter(U, sigma=sigma, mode='reflect')
    else:
        raise ValueError("method must be one of {'gaussian','uniform'}")

def plot_farm_deficit_map(wind_farm, x_resolution=300, y_resolution=100, z_resolution=100, save_path=None):
    if not wind_farm.turbines:
        print("No turbines found in wind farm.")
        return

    print("Generating Farm Deficit Map...")

    elevation_func = wind_farm.turbine_configs.elevation_func

    # 1. Define Visualization Domain
    all_x = [t.pos[0] for t in wind_farm.turbines]
    all_y = [t.pos[1] for t in wind_farm.turbines]
    D = wind_farm.turbines[0].D 
    Uhub_ref = wind_farm.turbines[0].Uhub 
    Zhub_ref = wind_farm.turbines[0].Zhub + wind_farm.turbines[0].pos[2]
    y_slice_val = wind_farm.turbines[0].pos[1]  # Cross-stream slice at T0 for side view
    z_min_plot = 0.0  # Minimum z for plotting ground

    min_wake_len = wind_farm.field_params.min_X * D
    max_wake_wid = wind_farm.field_params.max_Y * D
    max_wake_hgt = wind_farm.field_params.max_Z * D
    
    x_min = min(all_x) - 2 * D
    x_max = max(all_x) + min_wake_len
    y_min = min(all_y) - max_wake_wid
    y_max = max(all_y) + max_wake_wid
    z_min = 0.0
    z_max = max([(t.Zhub + t.pos[2]) for t in wind_farm.turbines]) + max_wake_hgt

    X_vis = np.linspace(x_min, x_max, x_resolution)
    Y_vis = np.linspace(y_min, y_max, y_resolution)
    Z_vis = np.linspace(z_min, z_max, z_resolution)

    Y_loc, Z_loc = np.meshgrid(Y_vis, Z_vis, indexing='ij')

    # 2. Calculate Background Flow
    fp = wind_farm.field_params
    z_safe = np.maximum(Z_loc, fp.z0 + 1e-3)
    U_in = fp.Uh * (np.log(z_safe / fp.z0) / np.log(fp.Zh / fp.z0))
    
    z_ref_idx = np.argmin(np.abs(Z_vis - Zhub_ref))
    y_ref_idx = np.argmin(np.abs(Y_vis - y_slice_val))

    # Initialize Maps
    U_xy_map = np.zeros((len(Y_vis), len(X_vis))) # (Y, X)
    U_xz_map = np.zeros((len(Z_vis), len(X_vis))) # (Z, X)

    # 3. Streamwise iteration. One marched field, so the velocity at each station is
    #    whatever the nearest upstream rotor carried there -- no combination step.
    for i, x_global in enumerate(X_vis):
        src = max((t for t in wind_farm.turbines if t.pos[0] < x_global),
                  key=lambda t: t.pos[0], default=None)
        if src is None:
            U_total_slice = U_in
        else:
            u_abs, _ = interpolate_local_velocity_field(
                src, x_global - src.pos[0], Y_loc, Z_loc, default=U_in)
            U_total_slice = np.asarray(u_abs)

        # Top View: Take all Y at fixed Z (Reference Hub Height)
        U_xy_map[:, i] = U_total_slice[:, z_ref_idx]
        
        # Side View: Take all Z at fixed Y (Centerline)
        U_xz_map[:, i] = U_total_slice[y_ref_idx, :]

    U_xy_map = smooth_2d(U_xy_map, method='gaussian')
    U_xz_map = smooth_2d(U_xz_map, method='gaussian')

    # 4. Plotting
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8)) # Added figsize

    # XY Grid
    X_grid_xy, Y_grid_xy = np.meshgrid(X_vis, Y_vis) 
    vmin, vmax = 0.2, 1.2
    levels = np.linspace(vmin, vmax, 50)
    
    im1 = ax1.contourf(X_grid_xy, Y_grid_xy, U_xy_map / Uhub_ref,
                       cmap='bwr', levels=levels, vmin=vmin, vmax=vmax, extend='both')

    # XZ Grid (Side view requires X and Z)
    X_grid_xz, Z_grid_xz = np.meshgrid(X_vis, Z_vis)

    im2 = ax2.contourf(X_grid_xz, Z_grid_xz, U_xz_map / Uhub_ref,
                       cmap='bwr', levels=levels, vmin=vmin, vmax=vmax, extend='both')

    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label(r'Normalized Velocity $U / U_{hub}$')
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.set_xlabel('Global Streamwise X (m)')
    ax1.set_ylabel('Global Cross-stream Y (m)')
    ax1.set_title(f'Top View (Z = {Zhub_ref:.1f}m)')

    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label(r'Normalized Velocity $U / U_{hub}$')
    cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_xlabel('Global Streamwise X (m)')
    ax2.set_ylabel('Global Vertical Z (m)')
    ax2.set_title('Side View (Y = 0m)')
    
    # --- TURBINE OVERLAYS ---
    for idx, t in enumerate(wind_farm.turbines):
        yaw_rad = np.deg2rad(t.yaw)
        
        # --- TOP VIEW (XY) ---
        # Draw a line representing the rotor diameter, rotated by yaw
        dx = (t.D / 2) * np.sin(yaw_rad)
        dy = (t.D / 2) * np.cos(yaw_rad)
        
        ax1.plot([t.pos[0] - dx, t.pos[0] + dx], 
                 [t.pos[1] + dy, t.pos[1] - dy], 
                 color='black', lw=3, solid_capstyle='round')
        
        ax1.text(t.pos[0], t.pos[1] + t.D * 0.6, f"T{idx}", color='white', 
                 ha='center', va='center', fontweight='bold',
                 path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])
        
        # --- SIDE VIEW (XZ) ---
        # 1. Draw the Tower (Vertical line from ground to hub)
        ax2.plot([t.pos[0], t.pos[0]], 
                 [t.pos[2], t.Zhub+t.pos[2]], 
                 color='black', lw=2)
        
        # 2. Draw the Rotor (Vertical line at hub height)
        z_top = t.pos[2] + t.Zhub + (t.D / 2)
        z_bot = t.pos[2] + t.Zhub - (t.D / 2)
        
        ax2.plot([t.pos[0], t.pos[0]], 
                 [z_bot, z_top], 
                 color='black', lw=4, solid_capstyle='round')

    # --- DRAW GROUND IN SIDE VIEW ---
    if elevation_func:
        # Calculate terrain line along the side-view slice
        z_ground_line = elevation_func(X_vis, np.full_like(X_vis, y_slice_val))
        ax2.fill_between(X_vis, z_min_plot, z_ground_line, color='#6d4c41', alpha=1.0, zorder=5) # Brown ground
        ax2.plot(X_vis, z_ground_line, color='black', lw=1, zorder=6)

    plt.tight_layout()

    if save_path:
        dir_name = save_path
        grid = wind_farm.get_grid()
        img_name = f"farm_map_{grid['rows']}x{grid['cols']}.png"
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig.savefig(os.path.join(dir_name, img_name), dpi=300, bbox_inches='tight')
        print(f"Saved farm map to {os.path.join(dir_name, img_name)}")
    else:
        plt.show()

def plot_turbine_layout(wind_farm, save_path=None):
    """Top-down farm layout: rotors colored by yaw (diverging, since yaw is signed),
    labeled with per-turbine efficiency."""
    turbines = wind_farm.turbines
    if not turbines:
        print("No turbines found in wind farm.")
        return

    positions = np.array([t.pos[:2] for t in turbines])
    yaws = np.array([float(t.yaw) for t in turbines])
    D = np.array([float(t.D) for t in turbines])
    eff = np.array([float(t.calculate_efficiency()) for t in turbines])

    yaw_abs_max = max(1.0, np.max(np.abs(yaws)))
    norm = TwoSlopeNorm(vmin=-yaw_abs_max, vcenter=0.0, vmax=yaw_abs_max)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 5))
    style_figure(fig)
    ax.set_aspect('equal')

    for (x, y), yaw, d, e in zip(positions, yaws, D, eff):
        color = plt.cm.RdBu_r(norm(yaw))
        circ = plt.Circle((x, y), d / 2, facecolor=color, edgecolor=INK_SECONDARY, linewidth=1.2, zorder=3)
        ax.add_patch(circ)
        ax.annotate(f"η={e:.2f}\nyaw {yaw:+.0f}°", (x, y + d / 2), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, color=INK_PRIMARY)

    pad = D.max() * 2
    ax.set_xlim(positions[:, 0].min() - pad, positions[:, 0].max() + pad)
    ax.set_ylim(positions[:, 1].min() - pad, positions[:, 1].max() + pad * 1.4)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Wind farm layout — yaw & efficiency by turbine', loc='left')
    style_axes(ax)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    cbar = fig.colorbar(mappable, ax=ax, label='Yaw angle (deg)')
    style_colorbar(cbar)

    fig.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, 'turbine_layout.png')
        fig.savefig(fname, dpi=300, bbox_inches='tight', facecolor=SURFACE)
        print(f"Saved turbine layout to {fname}")
    else:
        plt.show()

def plot_data(data, config, pause_interval=0.1, quiver_samples=35,
              show_streamwise=True, save_path=None, save_at_x=None, fps=10, dpi=150, show=True):
    """
    Main driver to visualize wake data. Coordinates setup, data prep, and animation loop.
    
    Parameters:
    -----------
    show : bool, optional (default=True)
        If True, displays the plots interactively. If False, only saves without displaying.
    """
    # Use non-interactive backend when not showing
    if not show and save_path:
        import matplotlib
        matplotlib.use('Agg')
    
    plt.close('all')

    all_u = [np.asarray(d.U) for d in data]
    all_omg = [np.asarray(d.OmegaX) for d in data]

    levels = {
        'u': np.linspace(0, 1, 21) * ceil(np.nanmax(all_u) / config.Uhub, 1),
        'omg': np.linspace(-1, 1, 21) * ceil(np.nanmax(np.abs(all_omg)) / (config.Uhub / config.D), 1)
        }
    
    # 1. Prepare common data and grids
    grid_info = _get_grid_info(data[0], config, config.Zhub)
    wake_history = _extract_streamwise_history(data, config, grid_info) if show_streamwise else None
    
    # 2. Setup Figure and Axes
    fig, axes_dict = _setup_layout(show_streamwise and wake_history is not None)
    
    # 3. Animation / Plotting Loop
    # Container for mutable state (colorbars) to avoid 'nonlocal' mess
    plot_state = {'cbar_vort': None, 'cbar_vel': None, 'cbar_wake': None}

    def update_frame(entry):
        _render_snapshot(axes_dict, plot_state, entry, config, grid_info, quiver_samples, levels)
        if wake_history:
            _render_streamwise(axes_dict, plot_state, entry, config, grid_info, wake_history)

    # Pre-identify indices for snapshots to save
    save_indices = {}
    if save_at_x is not None:
        actual_x = np.array([d.X / config.D for d in data])
        for target in save_at_x:
            # Find the index of the data entry closest to the user's target x/D
            idx = np.argmin(np.abs(actual_x - target))
            save_indices[idx] = target

    # Execution
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        _save_animation(fig, data, update_frame, save_path, config.yaw, fps, dpi, pause_interval if show else 0, save_indices)
    elif show:
        _show_live(data, update_frame, pause_interval)

    return fig, axes_dict

def _get_grid_info(data_snapshot, config, z_target):
    """Normalizes grids and finds indices for hub-height slicing."""
    yloc = np.asarray(data_snapshot.yloc) / config.D
    zloc = np.asarray(data_snapshot.zloc) / config.D
    
    # Find indices closest to Center/Hub
    z_target = z_target / config.D
    y_center_idx = yloc.shape[0] // 2
    z_hub_idx = np.argmin(np.abs(zloc[0, :] - z_target))
    
    return {
        'yloc': yloc, 'zloc': zloc,
        'y_center_idx': y_center_idx, 'z_hub_idx': z_hub_idx,
        'Ny': yloc.shape[0], 'Nz': yloc.shape[1]
    }

def _extract_streamwise_history(Data, config, grid):
    """Pre-processes the full dataset to extract streamwise evolution arrays."""
    X_pos, U_hub_profiles = [], []
    
    for d in Data:
        X_pos.append(d.X / config.D)
        # Extract entire Y-profile at hub height Z
        U_hub_profiles.append(np.asarray(d.U)[:, grid['z_hub_idx']])
            
    if len(X_pos) < 2: return None

    # Create meshgrid for the bottom-right heatmap
    X_grid, Y_grid = np.meshgrid(X_pos, grid['yloc'][:, 0])
    
    return {
        'X': np.array(X_pos),
        'U': np.array(U_hub_profiles).T, # Shape: (Ny, N_snapshots)
        'X_grid': X_grid,
        'Y_grid': Y_grid
    }

def _setup_layout(has_streamwise):
    """Creates the figure and returns a labeled dictionary of axes."""
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

    if has_streamwise:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        return fig, {'vort': axes[0,0], 'vel': axes[0,1], 'prof': axes[1,0], 'wake': axes[1,1]}
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        return fig, {'vort': axes[0], 'vel': axes[1]}

def _render_snapshot(axes, state, entry, config, grid, q_samples, levels):
    """Handles the top row: Contour plots for current timestep."""
    # 1. Vorticity (Top Left)
    ax_vort = axes['vort']
    
    omg = np.asarray(entry.OmegaX)
    norm_factor = config.Uhub / config.D
    omg_norm = omg / norm_factor
    
    if 'cf_vort' not in state:
        ax_vort.set_aspect('equal', 'box')
        state['cf_vort'] = ax_vort.contourf(grid['yloc'], grid['zloc'], omg_norm,
                        levels=levels['omg'], cmap='RdBu_r', extend='both')
        state['title_vort'] = ax_vort.set_title(f'$\\Omega_x$ (Normalized) at X/D = {entry.X/config.D:.1f}')
        state['cbar_vort'] = ax_vort.figure.colorbar(state['cf_vort'], ax=ax_vort, label='Vorticity')
        state['cbar_vort'].ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        state['cf_vort'].remove()
        state['cf_vort'] = ax_vort.contourf(grid['yloc'], grid['zloc'], omg_norm,
                                    levels=levels['omg'], cmap='RdBu_r', extend='both')
        state['title_vort'].set_text(f'$\\Omega_x$ (Normalized) at X/D = {entry.X/config.D:.1f}')
        
    # Quiver overlay
    state = _add_quiver(ax_vort, entry, state, grid, q_samples)

    # 2. Velocity (Top Right)
    ax_vel = axes['vel']

    u_def_norm = np.asarray(entry.U) / config.Uhub

    if 'cf_vel' not in state:
        ax_vel.set_aspect('equal', 'box')
        state['cf_vel'] = ax_vel.contourf(grid['yloc'], grid['zloc'], u_def_norm, levels=levels['u'], cmap='turbo')
        state['title_vel'] = ax_vel.set_title(f'U/Uhub at X/D = {entry.X/config.D:.1f}')
        state['cbar_vel'] = ax_vel.figure.colorbar(state['cf_vel'], ax=ax_vel, label='Normalized Streamwise Velocity')
        state['cbar_vel'].ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        state['cf_vel'].remove()
        state['cf_vel'] = ax_vel.contourf(grid['yloc'], grid['zloc'], u_def_norm, levels=levels['u'], cmap='turbo')
        state['title_vel'].set_text(f'U/Uhub at X/D = {entry.X/config.D:.1f}')

def _add_quiver(ax, entry, state, grid, samples):
    """Adds quiver arrows to an existing axis."""
    dN = max(1, int(grid['Ny'] / samples))
    sl = np.s_[::dN, ::dN] # Slice object

    V_new = np.asarray(entry.V)[sl]
    W_new = np.asarray(entry.W)[sl]

    if 'quiver' not in state:
        state['quiver'] = ax.quiver(grid['yloc'][sl], grid['zloc'][sl],
                V_new, W_new, color='k', scale=16.0, angles='xy', zorder=2)
    else:
        state['quiver'].set_UVC(V_new, W_new)
    return state

def _render_streamwise(axes, state, entry, config, grid, history):
    """Handles bottom row: Profiles and Heatmap."""
    current_x = entry.X / config.D
    
    # 3. Radial Profiles (Bottom Left)
    ax_prof = axes['prof']
    
    if 'prof_lines' not in state:
        # Plot background profiles (static snapshots)
        num_profs = min(5, len(history['X']))
        indices = np.linspace(0, len(history['X'])-1, num_profs, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, num_profs))
        
        u_norm = history['U'] / config.Uinf
        for i, idx in enumerate(indices):
            ax_prof.plot(grid['yloc'][:,0], u_norm[:, idx], 
                    color=colors[i], marker='o', ms=3, alpha=0.5, label=f'X/D={history["X"][idx]:.1f}')
            
        # Highlight current profile
        curr_idx = np.argmin(np.abs(history['X'] - current_x))
        state['prof_lines'] = ax_prof.plot(grid['yloc'][:,0], u_norm[:, curr_idx], 'r-', lw=3, label='Current Profile')[0]
        
        ax_prof.set_title('Hub Height Velocity Profiles')
        ax_prof.set_ylabel('U/Uhub')
        ax_prof.set_xlabel('Normalized Cross-stream Distance Y/D')
        ax_prof.grid(True, alpha=0.3)
        ax_prof.legend(fontsize=8)
    else:
        curr_idx = np.argmin(np.abs(history['X'] - current_x))
        u_norm = history['U'] / config.Uinf
        state['prof_lines'].set_ydata(u_norm[:, curr_idx])

    # 4. Wake Evolution Heatmap (Bottom Right)
    ax_wake = axes['wake']
    
    if 'v_line' not in state:
        u_norm = history['U'] / config.Uinf
        mesh = ax_wake.pcolormesh(history['X_grid'], history['Y_grid'], u_norm, cmap='turbo', shading='auto', rasterized=True)
        state['v_line'] = ax_wake.axvline(current_x, color='r', ls='--')
        ax_wake.set_title('Wake Evolution (Top Down)')
        ax_wake.set_xlabel('Streamwise Distance X/D')
        ax_wake.set_ylabel('Normalized Cross-stream Distance Y/D')
        state['cbar_wake'] = ax_wake.figure.colorbar(mesh, ax=ax_wake, label='Normalized Velocity U/Uhub')
    else:
        state['v_line'].set_xdata([current_x, current_x])

def _save_animation(fig, data, update_func, path, yaw, fps, dpi, interval, save_indices):
    """Handles the video writing logic."""
    folder_name = f"Yaw{yaw:.2f}"
    os.makedirs(os.path.join(path, folder_name), exist_ok=True)
    try:
        out_path = os.path.join(path, folder_name, f"Yaw{yaw:.2f}_Wake.gif")
        writer = PillowWriter(fps=fps)
        with writer.saving(fig, out_path, dpi):
            for i, entry in enumerate(data):
                update_func(entry)
                fig.canvas.draw()

                if i in save_indices.keys():
                    fname = os.path.join(path, folder_name, f"Wake_xD_{save_indices[i]:.1f}.png")
                    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
                    print(f"Saved snapshot at x/D = {save_indices[i]:.1f}")

                writer.grab_frame()
                if interval > 0:  # Only pause if interval is set (i.e., when showing)
                    plt.pause(interval)
    except Exception as e:
        print(f"Error saving animation: {e}")

def _show_live(data, update_func, interval):
    """Handles standard live plotting."""
    for entry in data:
        update_func(entry)
        plt.pause(interval)

def npy_to_list(d):
    if isinstance(d, dict):
        return {k: npy_to_list(v) for k, v in d.items()}
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d

def ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)