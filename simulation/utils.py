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

# Exponent of the p-norm combining added-turbulence contributions from several upstream
# rotors, Li, Wang, Dong, Yang & Zhang (2023) Energy 276, 127491 Eq (22). m = 1 is the
# linear sum, m -> inf the dominant-upstream rule.
#
# We use m = 2, NOT the m = 2.5 Li fits for aligned rows. m = 2 adds the contributions in
# variance, which is what independent fluctuations do, and is the only value in the family
# with a physical derivation rather than a fit. m is a level knob -- its optimum moves with
# the dataset and often sits on a sweep boundary -- so it is fixed on principle and never
# tuned, and a sweep over it is not a validation.
CRESPO_M = 2.0


def NuT_model(wake_field, config, I_amb, up_a, up_D, up_pos_x, up_pos_y, up_Uhub,
              up_mask):
    """
    Turbulent eddy viscosity closure for a single turbine's wake, following the
    piecewise-linear model of Du, Zhu, Li, Ge, Li & Liu (2025), "Modeling of
    turbulence kinetic energy added by wind-turbine wakes in the atmospheric
    boundary layer", arXiv:2511.19881, Eq. (26):

        NuT(x) / (U0*D) = (kv1*TI - kv0) * x/D,      x/D <= xv1/TI
                        = (kv1*TI - kv0) * xv1/TI,   x/D >  xv1/TI

    with kv1=0.05, kv0=0.001, xv1=0.5 (paper's fitted constants). Extended here
    to multi-turbine wakes by accumulating turbulence intensity across upstream
    wakes via the Crespo & Hernandez (1996) added-turbulence correlation, and by
    measuring x/D relative to 'config' itself (whose wake is being marched) --
    the paper's model was validated only for a stand-alone turbine with a fixed
    ambient TI, so this multi-turbine accumulation is an extension, not something
    the source paper tested directly.
    """
    kv1, kv0, xv1 = 0.05, 0.001, 0.5
    IU_TO_TI = 1.28  #(1.28) Du et al. (2025), Sec 2.3: Iu ~ 1.28 * TI under neutral conditions
    m = CRESPO_M

    x_global = config.pos[0] + wake_field.X

    x_D_up = (x_global - up_pos_x) / up_D
    # Crespo & Hernandez (1996) JWEIA 61, 71-85, valid for 5 < x/D < 15, 0.07 < I0 < 0.14,
    # 0.1 < a < 0.4.
    delta_I = 0.73 * up_a ** 0.8325 * I_amb ** (-0.0325) * x_D_up ** (-0.32)
    contribution = (delta_I * up_Uhub / config.U0) ** m

    # TI RESOLVED ACROSS THE PLANE, not averaged over it.
    #
    # An upstream rotor raises the turbulence where ITS wake is, and the lateral extent of
    # that influence is the |dy| < 3D criterion this model already used -- it was simply
    # evaluated once, at the marching rotor, and the single resulting value was applied to
    # every wake on the cross-plane. With one turbine per streamwise station that is
    # exactly right, so single rows are unaffected. With several it is not: the plane
    # inherits whichever rotor happened to lead the station, and since the lead is just the
    # first to sort there, two identical farms listed in a different order disagreed by up
    # to 13.6% in rotor inflow. Evaluating the same criterion per grid column removes that
    # without changing the correlation, its constants, or the accumulation rule.
    #
    # NuT therefore returns a FIELD. Its lateral variation sits at |dy| = 3D from a rotor,
    # where the deficit gradient is negligible (measured: the transition occupied 2 of 140
    # columns and overlapped none of the 69 in-wake ones), so nu_T may stay outside the
    # divergence in model_solver -- the grad(nu_T).grad(U - Uin) term it would otherwise
    # need is zero wherever it would act.
    y_rel = (wake_field.yloc + config.pos[1])[..., None] - up_pos_y    # (ny, nz, n_up)
    qualifies = up_mask & (x_D_up >= 3.0) & (jnp.abs(y_rel) < 3.0 * up_D)
    Iu_sq = I_amb ** m + jnp.sum(jnp.where(qualifies, contribution, 0.0), axis=-1)

    TI_total = (Iu_sq ** (1 / m)) / IU_TO_TI
    x_D = wake_field.X / config.D
    x_D_threshold = xv1 / TI_total
    coeff = jnp.maximum(kv1 * TI_total - kv0, 0.0)

    return coeff * jnp.minimum(x_D, x_D_threshold) * config.U0 * config.D

#: Du et al. (2025) Eq (26) eddy viscosity and Eq (28) dissipation length group, both as
#: functions of a turbulence intensity and a shear-layer age. Kept in one place so the
#: wake-added TKE budget and the deficit's own nu_T cannot drift apart.
DU_KV1, DU_KV0, DU_XV1 = 0.05, 0.001, 0.5
IU_TO_TI = 1.28   # Du et al. Sec 2.3: Iu ~ 1.28 TI under neutral conditions


def du_nu_t(TI, age_D, U0, D):
    """Du Eq (26): nu_t/(U0 D) = (kv1 TI - kv0) * min(age/D, xv1/TI)."""
    return (jnp.maximum(DU_KV1 * TI - DU_KV0, 0.0)
            * jnp.minimum(age_D, DU_XV1 / TI) * U0 * D)


def du_psi(TI, age_D, D):
    """Du Eq (28): Psi/D^2 = 0.67 (0.2 TI + 0.015)^2 / (0.4 TI + 0.010) * age/D.

    Psi = c l_m^2 / C_eps sets the dissipation timescale Psi/nu_t, and because l_m is the
    mixing length it grows with the age of the SHEAR LAYER -- which restarts at every
    rotor. Referencing it to the farm inlet instead makes the amplitude (~sqrt(Psi)) grow
    without bound down a row: measured on the three-turbine experiment, the peak I_u error
    went -1% in bay 1 to +20% and +28% in bays 2 and 3.
    """
    return (0.67 * (0.2 * TI + 0.015) ** 2 / (0.4 * TI + 0.010)
            * jnp.maximum(age_D, 1e-6) * D ** 2)


#: Du et al. (2025) Sec 3.2: nu_T = c <k_w>^(1/2) l_m, with c = 0.67 the mean across his
#: calibration cases.
DU_C_NU = 0.67

#: Standard Prandtl-Kolmogorov constant, nu_T = c_mu k^2/eps (Launder & Spalding 1974).
C_MU = 0.09

#: Bradshaw's structure parameter a1 = -<uv>/k. NOT an independent constant: consistency
#: between nu_T = c sqrt(k) l_m, eps = c_eps k^(3/2)/l_m and nu_T = c_mu k^2/eps forces
#: c*c_eps = c_mu, and the equilibrium length l_m = sqrt(c_eps/c) sqrt(k)/S then gives
#: nu_T = sqrt(c*c_eps) k/S = sqrt(c_mu) k/S, independent of c. So a1 = sqrt(c_mu) = 0.30 --
#: the textbook identity behind Menter's SST shear-stress limiter. Bradshaw's own measured
#: value across boundary layers and free shear layers is 0.31, which agrees to 3%; that
#: agreement is a cross-check on Du's c = 0.67 with the standard c_mu, not a second fit.
BRADSHAW_A1 = C_MU ** 0.5

#: eps = c_eps k^(3/2)/l_m, determined by c*c_eps = c_mu rather than fitted.
DU_C_EPS = C_MU / DU_C_NU


#: von Karman constant, for the Prandtl-Blackadar ABL mixing length l = kappa*z/(1 + kappa*z/l0),
#: which tends to kappa*z near the surface. Used to BOUND the wake mixing length -- see
#: ABL_LENGTH_CAP in turbine_physics.
KARMAN = 0.4


#: l_m as a fraction of the wake width. Klemmer & Howland (2025) JRSE 17, 033304 take l_m to
#: be the wake width itself with their own coefficient; 0.08 is the plane-mixing-layer value
#: and, measured on the full-wake set, it reproduces Du's own l_m at x = 3D to three figures
#: (0.0776 vs 0.0775), so it is not an extra fit.
LM_OVER_WIDTH = 0.08


def wake_width(Uw, yloc, zloc):
    """Equivalent top-hat DIAMETER of the deficit: A = int Uw dA / max(Uw), W = 2 sqrt(A/pi).

    Unlike an age, this does not restart at a downstream rotor -- T2's fresh top-hat is added
    to an already-wide profile, so W is continuous across a station boundary. Measured through
    the three-turbine row: 0.978 at the end of bay 1, 1.049 at T2's rotor, 1.165, then 1.196 at
    T3's rotor and 1.290. Also unlike the vorticity thickness, which a freshly seeded top-hat
    pins through its max gradient.
    """
    dy = yloc[1, 0] - yloc[0, 0]
    dz = zloc[0, 1] - zloc[0, 0]
    Uw = jnp.maximum(Uw, 0.0)
    A = jnp.sum(Uw) * dy * dz / (jnp.max(Uw) + 1e-9)
    return 2.0 * jnp.sqrt(jnp.maximum(A, 0.0) / jnp.pi)


def width_equivalent_age(Uw, yloc, zloc, TI, D):
    """The age whose Du l_m equals LM_OVER_WIDTH * W, so Du's formulae are reused unchanged."""
    W = wake_width(Uw, yloc, zloc)
    return LM_OVER_WIDTH * (W / D) / (0.2 * TI + 0.015)


def du_mixing_length(TI, age_D, D):
    """Du's algebraic l_m = (0.2 TI + 0.015) * age, the k_lm implied by his Eq (28).

    Used ONLY by the TKE budget. The budget cannot use the equilibrium length that
    nu_t_from_tke uses: substituting it makes production and dissipation identically equal
    for any k and S --

        P   = nu_t S^2       = c sqrt(k) * sqrt(c_eps/c) sqrt(k)/S * S^2 = sqrt(c c_eps) k S
        eps = c_eps k^(3/2)/l_m                                          = sqrt(c c_eps) k S

    -- so the source vanishes, k_w stays at its seeded zero forever and the closure collapses
    to the ambient. A length defined BY local equilibrium cannot also generate departures
    from it, which is exactly why Du carries a separate Psi. Making the two length scales one
    requires transporting a second variable (k-eps, or Klemmer & Howland's k-l), not
    substituting one into the other.
    """
    return (0.2 * TI + 0.015) * age_D * D


#: Fraction of the ambient LATERAL velocity rms surviving the DWM low-pass filter at
#: f = U/(2D) (Larsen et al. 2008), i.e. streamwise wavelength 2D under Taylor's hypothesis.
#: MEASURED from data/ABL_flow_3turbine.h5 at |y| > 220 m, outside the wakes: variance
#: fraction 0.544, so phi = 0.738. Range 0.738-0.762 over C_meander = 1.9-2.1; 0.671 at the
#: D_m mean of 2.5D.
PHI_MEANDER = 0.738

#: Lateral-to-streamwise turbulence intensity ratio. MEASURED in the same sample as 0.735
#: (I_u = 0.0742, I_v = 0.0546); the usual ABL default is 0.8.
I_V_OVER_IU = 0.735


def meander_diffusivity(I_amb, age_D, U0, D):
    """Apparent diffusivity of the time average produced by wake meandering.

    A meandering wake is a rigid deficit whose centre wanders, so the TIME AVERAGE is exactly
    a convolution of the instantaneous deficit with the displacement PDF:
    <Uw>(y) = int p(eta) Uw(y-eta) deta. For Gaussian p of variance sigma_m^2 that is a
    Gaussian smoothing of width sigma_m, and a diffusion operator acting over distance x at
    speed U smooths by variance 2 nu x / U. Matching the GROWTH of the two,

        nu_m = (U/2) d(sigma_m^2)/dx ,    sigma_m = I_vf x   (frozen advection)
             = U I_vf^2 x ,               I_vf = PHI_MEANDER * I_V_OVER_IU * I_u

    Both the convolution and the diffusion conserve int Uw dy, which is the quantity the wind
    tunnel and the LES agree on while disagreeing 20% on width -- see
    [[les-wake-too-narrow-meandering]].

    THREE LIMITS, all stated rather than hidden:
      * Correct for the mean velocity, NOT for power: a rotor reads <U^3> and meandering gives
        <U^3> != <U>^3, so expect field R2 to gain more than power.
      * Double-counting risk: Du's Psi constants were calibrated on time-averaged LES TKE,
        which already contains meandering. The defence is empirical -- if it were double
        counting, a multiplier of 1.0 would already be optimal, and the measured optimum is
        1.6.
      * age_D is the STATION-LOCAL age on purpose. sigma_m ~ x is unbounded, which is exactly
        what made a farm-referenced l_m diverge on 8 turbines; the local age is bounded by
        construction and is right for the freshly shed deficit that dominates each bay, while
        under-counting the carried residual.
    """
    i_vf = PHI_MEANDER * I_V_OVER_IU * I_amb
    return U0 * i_vf ** 2 * jnp.maximum(age_D, 0.0) * D


def ambient_tke(I_amb, U0):
    """k_B from TI = sqrt(2k/3)/U0, with TI the ambient value Iu/1.28."""
    return 1.5 * ((I_amb / IU_TO_TI) * U0) ** 2


def nu_t_budget(kw, I_amb, age_D, U0, D):
    """The TKE budget's own nu_t, in the same Prandtl-Kolmogorov form as the deficit's.

    Replaces Du Eq (26) here. Eq (26) takes the AMBIENT turbulence intensity, so using it for
    the budget was defensible, but it left an algebraic correlation and its three fitted
    constants inside an otherwise transport-based closure, and it is the one piece of the
    model applied outside its calibration range elsewhere. c sqrt(k) l_m uses the same c and
    the same k the rest of the closure uses; only l_m differs, and it has to -- see
    du_mixing_length.
    """
    k = jnp.maximum(ambient_tke(I_amb, U0) + kw, 0.0)
    return DU_C_NU * jnp.sqrt(k) * du_mixing_length(I_amb / IU_TO_TI, age_D, D)


def nu_t_from_tke(kw, I_amb, U0, Uw, yloc, zloc):
    """Eddy viscosity from the TKE field, Du Sec 3.2, rather than from his Eq (26).

    Eq (26) takes the AMBIENT turbulence intensity as its input and is calibrated over
    TI = 0.041-0.071 (his Table 2). Feeding it the LOCAL in-wake TI puts it at TI ~ 0.117 in
    the shear layers -- 1.7x past that maximum -- and since it is linear in TI with no
    saturation it simply keeps growing: measured 2.8x Crespo's nu_T there, which recovers
    T1's wake too fast and over-predicts T2's power by 45%. nu_T = c sqrt(k_B + k_w) l_m
    instead uses k_w as the turbulence kinetic energy it is and stays inside the regime each
    piece was built for.

    The length scale is fixed by local equilibrium rather than by a wake age. Equating
    production nu_t S^2 to dissipation c_eps k^(3/2)/l_m inside nu_t = c sqrt(k) l_m gives
    l_m = sqrt(c_eps/c) sqrt(k)/S, i.e. nu_t = a1 k/S -- Bradshaw's structure-parameter
    relation. So both scales come from fields the march already solves, and no reference to
    any rotor's x is needed.

    That reference is what the age proxies could not supply down a row. Du's
    l_m/D = (0.2 TI + 0.015) x/D describes a single isolated rotor; taken from each station
    it restarts the mixing length at zero although T2's shear layer is already ~1D wide, and
    taken from the farm inlet it grows without bound. Here l_m is read AT the point of
    maximum shear, so a rotor entering an already turbulent field finds a large k there as
    well as a large S, instead of being reset. Reading both fields at that one point also
    keeps the ratio finite, since S vanishes on the wake axis and in the free stream.

    Measured against Eq (26) on the full-wake set (cases C/A/B): bay-1 bias
    +17.8/+6.7/+9.4 -> +4.9/+2.7/+5.6 and T2 power error +45.1/+18.1/+28.6 -> +1.0/-1.6/+6.0.
    It does not move bays 2-3, and that is not its job -- see NUT_FROM_TKE in
    turbine_physics.
    """
    k = jnp.maximum(ambient_tke(I_amb, U0) + kw, 0.0)
    dWdY, dWdZ = jnp.gradient(Uw, yloc[:, 0], zloc[0, :])
    S = jnp.sqrt(dWdY ** 2 + dWdZ ** 2)
    i = jnp.argmax(S)
    l_m = BRADSHAW_A1 / DU_C_NU * jnp.sqrt(jnp.ravel(k)[i]) / (jnp.ravel(S)[i] + 1e-9)
    return DU_C_NU * jnp.sqrt(k) * l_m


def advance_kw(kw, U, Uin, yloc, zloc, dx, nu_t, psi, U0):
    """One march step of Du et al. (2025) Eq (13), the wake-added TKE budget:

        U0 dk_w/dx = nu_t grad^2 k_w - (nu_t/Psi) k_w + nu_t |grad_perp U|^2

    i.e. cross-plane diffusion, dissipation, and shear production. Du's Eq (14) is the
    analytic solution of this same equation; marching it instead avoids that solution's
    modified-Bessel approximation, its unreconciled Psi vs psi, and the undefined delta in
    its integration limit -- and because the cross-plane here is genuinely two-dimensional,
    the operator is just the 2-D Laplacian, with no axis singularity and NO axisymmetry
    assumption. Du's own formulation is axisymmetric in <k_w>_theta and therefore cannot
    represent the unequal lateral peaks the measurements show; this can.

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
    d2y = jnp.gradient(jnp.gradient(kw, yloc[:, 0], axis=0), yloc[:, 0], axis=0)
    d2z = jnp.gradient(jnp.gradient(kw, zloc[0, :], axis=1), zloc[0, :], axis=1)
    rhs = nu_t * (d2y + d2z) - (nu_t / psi) * kw + prod
    return jnp.maximum(kw + (dx / U0) * rhs, 0.0)


def advance_kwa(kwa, kw, yloc, zloc, dx, nu_t, psi, U0, prod=None):
    """One march step of the TRANSPORTED turbulence age, carried as M = k_w * A.

        U0 dM/dx = nu_t grad^2 M - (nu_t/Psi) M + U0 k_w

    The first two terms are the k_w operators: diffusion and dissipation carry the age with
    the turbulence they move or destroy. The last term is the ageing itself: with no other
    term, dM/dx = k_w, i.e. dA/dx = 1, so every unit of k_w gets one unit older per unit x.

    `prod` decides what age newly generated turbulence carries, and the choice matters a lot:

      * prod = None -> age ZERO. Correct for a TRACER age, and it dilutes the mean age at
        every rotor. Measured, A reaches only 3.56 D at a farm distance of 14 D, because the
        layer is continuously regenerated, so the station-age floor dominates and the result
        is barely better than the station-local reference.
      * prod = P    -> the LOCAL age, by adding A*P. Then dA/dx = 1 exactly, independent of
        production, so A is the distance the LAYER has travelled, accumulated through the
        field rather than read off rotor positions.

    The second is right for a LENGTH scale: l_m is a property of the shear layer, and
    turbulence generated inside an already-thick layer is generated at that layer's scale,
    not at zero scale. Only a passive tracer would start at zero.

    WHY THIS EXISTS. Du's l_m = k_lm x needs an x, and reading it off rotor positions is
    wrong in two different ways: the station-local age restarts the eddy size at every rotor
    (eddies do not shrink when they pass a turbine), while a farm-referenced age is a
    streamwise distance irrespective of lateral position, so a rotor off to the side and
    upstream rewrote a column it never touched -- measured at 12.26%, worse than the 9.325%
    that disqualified Crespo. A is LOCAL by construction: turbulence that never reaches a
    column cannot contribute to the age there.

    Returns M; the age is A = M/k_w, so read it through `age_from_kwa`.
    """
    d2y = jnp.gradient(jnp.gradient(kwa, yloc[:, 0], axis=0), yloc[:, 0], axis=0)
    d2z = jnp.gradient(jnp.gradient(kwa, zloc[0, :], axis=1), zloc[0, :], axis=1)
    rhs = nu_t * (d2y + d2z) - (nu_t / psi) * kwa + U0 * kw
    if prod is not None:
        rhs = rhs + (kwa / jnp.maximum(kw, 1e-9)) * prod
    return jnp.maximum(kwa + (dx / U0) * rhs, 0.0)


def age_from_kwa(kwa, kw, D, floor_D, cap_D):
    """Mean turbulence age in diameters, A = M/k_w, clipped to [floor_D, cap_D].

    The floor keeps the fresh near wake on Du's own short-age branch where k_w is still tiny
    and A is noise; the cap is the saturation discussed under BUDGET_AGE_CAP.
    """
    A = kwa / jnp.maximum(kw, 1e-9) / D
    return jnp.clip(jnp.where(kw > 1e-8, A, floor_D), floor_D, cap_D)


def ti_from_kw(kw, I_amb, U0):
    """Total turbulence intensity from ambient plus marched wake-added TKE.

    TI = sqrt(2 k/3)/U0 (Du Sec 2.3), so the added part combines with the ambient in
    quadrature exactly as the old p-norm did at m = 2 -- but the added part now comes from
    transport rather than from a correlation.
    """
    iu_added = IU_TO_TI * jnp.sqrt(2.0 / 3.0 * kw) / U0
    return jnp.sqrt(I_amb ** 2 + iu_added ** 2) / IU_TO_TI


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