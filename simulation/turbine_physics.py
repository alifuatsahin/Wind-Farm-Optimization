import dataclasses
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import erf as jerf

from .turbine_state import LocalConditions, VortexSimConfig, DeficitFieldConfig, pack_upstream_turbines
from .utils import (smooth_2d, NuT_model, du_nu_t, du_psi, advance_kw, ti_from_kw,
                    nu_t_from_tke, nu_t_budget, ambient_tke, width_equivalent_age,
                    LM_OVER_WIDTH, meander_diffusivity, KARMAN,
                    advance_kwa, age_from_kwa)
from .vortex_model import _simulate_vortex_evolution_jit, _UNSTACKED
from .model_solver import advance_wake_field
from .field_ops import interpolate_vec_data


def init_Uin(params):
    zsafe = np.maximum(params.zloc, params.field_params.z0 + 1e-6)  # avoid log(0) issues
    Uin = params.Uh * (np.log(zsafe / params.field_params.z0) / np.log(params.Zh / params.field_params.z0))
    return Uin


def nominal_hub_velocity(params):
    """Rotor-averaged undisturbed inflow -- depends only on config, never on the mutated Uin.
    """
    Uin = init_Uin(params)
    r2 = (((params.yloc + params.Yoffset) ** 2) / (np.cos(params.beta) ** 2)
          + (params.zloc - params.Zhub) ** 2)
    return float(np.mean(Uin[np.sqrt(r2) <= params.R]))


def init_local_conditions(params):
    return LocalConditions(
        Uhub=nominal_hub_velocity(params),
        V=np.zeros_like(params.yloc),
        W=np.zeros_like(params.zloc),
        Uin=init_Uin(params),
    )


def compute_omega(params, Uhub):
    return params.TSR * Uhub / params.R


def compute_gamma0(params, Uhub):
    """Total circulation shed from the blade tips, Zong & Porte-Agel (2020) Eq (4.3):
    Gamma0 = k*pi*Uh^2*Ct / (Omega*(1+a')), with a' dropped at large tip-speed ratio
    and k applied in compute_dgamma.

    Ct is the thrust coefficient AT the yaw angle. Eq (4.3) comes from vortex-cylinder
    theory relating shed circulation to thrust, and a yawed rotor produces less thrust
    and less bound circulation."""
    omega = compute_omega(params, Uhub)
    return np.pi * Uhub ** 2 * params.Ct_yawed / omega


def compute_Ut(params, Uhub, beta=None):
    """Blade-tip velocity around the azimuth. `beta` defaults to the turbine's own yaw;
    pass 0.0 to get the same rotor unyawed, which compute_dgamma uses as a reference."""
    beta = params.beta if beta is None else beta
    omega = compute_omega(params, Uhub)
    z = params.Zhub + params.D / 2.0 * np.sin(params.phi)
    zsafe = np.maximum(z, params.field_params.z0 + 1e-6)
    Utbl = params.Uh * (np.log(zsafe / params.field_params.z0) / np.log(params.Zh / params.field_params.z0))
    cb = np.cos(beta)
    a = params.a_at(beta)
    U_tx = (1 - a) * Utbl - omega * params.R * np.sin(params.phi) * np.sin(beta)
    U_ty = -omega * params.R * np.sin(params.phi) * cb
    U_tz = omega * params.R * np.cos(params.phi)
    return np.array([U_tx, U_ty, U_tz]).T


def _shed_shape(params, Uhub, beta):
    """Unnormalised azimuthal distribution of shed circulation, Zong Eq (4.4)."""
    Ut = compute_Ut(params, Uhub, beta)
    alpha = np.arcsin(Ut[:, 0] / np.sqrt(np.sum(Ut ** 2, axis=1)))
    return np.sin(alpha) * params.dphi


#: Scaling on the trailing (root-vortex) circulation in Zong & Porte-Agel (2020) Eq (4.3),
#: Gamma_0 = k*pi*Uh^2*Ct/Omega. 0.45 is Zong's value and it is MEASURED, not fitted: his
#: figure 4(a) compares the azimuthal circulation distribution from stereo-PIV against
#: Eq (4.4) at beta = -30, 0 and +30 deg, and states "the experimental circulation values
#: are only 45% of the modelled values".
#:
#: It has a physical meaning. Eq (4.3) comes from vortex-cylinder theory assuming zero drag
#: and uniform spanwise circulation. Real blades violate both: part of the thrust is
#: drag-induced and generates no circulation, and non-uniform spanwise loading sheds some
#: vorticity inboard rather than at the tip. The paper's appendix says k is therefore the
#: ratio of lift-induced thrust to total thrust, obtainable per turbine from blade element
#: momentum theory -- so it is a per-rotor physical parameter, not a universal constant.
#:
#: THIS WAS SET TO 1.0 AND THAT WAS WRONG. The reasoning was that Bastankhah et al. (2022)
#: JFM 933 A2 Eq (38) derives Gamma_r = (pi R/lambda) U_h C_T cos^2(beta) parameter-free,
#: which reduces exactly to compute_gamma0 and so implies k = 1. The algebra is right but
#: the conclusion is not: Eq (38) is the same IDEALISED vortex-cylinder value as Zong's
#: Eq (4.3) with k = 1, and Zong measured the real rotor to shed 45% of it. Deriving the
#: ideal and measuring the actual are not in conflict -- k is the gap between them.
#:
#: Measured consequences of k = 1, all bad: the yawed wake width ratio ran past the
#: unyawed one by 4D (1.036, unphysical), and the curl at 6D fell from 0.134 to 0.098 D
#: against an LES value of 0.318 D.
GAMMA_TRAILING_K = 0.45


def compute_dgamma(params, Uhub):
    """Shed circulation per azimuthal element, with the torque- and force-driven parts
    scaled separately.
    """
    gamma_ref = -compute_gamma0(params, Uhub) * GAMMA_TRAILING_K

    shape_b = _shed_shape(params, Uhub, params.beta)
    shape_0 = _shed_shape(params, Uhub, 0.0)
    dgamma_b = gamma_ref / np.sum(shape_b) * shape_b
    dgamma_0 = gamma_ref / np.sum(shape_0) * shape_0

    # zero-mean, so rescaling it leaves the total (torque) circulation untouched
    asym_yaw = (dgamma_b - dgamma_b.mean()) - (dgamma_0 - dgamma_0.mean())
    z_rel = params.R * np.sin(params.phi)
    M_yaw = float(np.sum(asym_yaw * z_rel))
    M_target = (0.5 * Uhub * (np.pi * params.D ** 2 / 4.0) * params.Ct
                * np.sin(params.beta) * np.cos(params.beta) ** 2)
    if abs(M_yaw) > 1e-12:
        asym_yaw = asym_yaw * (M_target / M_yaw)
    return dgamma_0 + asym_yaw


def calculate_efficiency(params, Uhub):
    P = (Uhub ** 3) * params.Cp
    nominal_P = (nominal_hub_velocity(params) ** 3) * params.config.Cp
    return P / nominal_P


def abl_nu_slope(params):
    """d(nu_T)/dx for a vortex in the ABL -- Shapiro, Gayme & Meneveau (JFM), "Generation
    and decay of counter-rotating vortices downstream of yawed wind turbines in the
    atmospheric boundary layer", Eq (4.7):

        nu_T(x) = u* * 2k(x - x0)/sqrt(24),   k = kappa/ln(z_h/z0),   u* = k*U_inf

    Their argument (Sec 4): a vortex in the ABL is not diffused by its own swirl but by
    boundary-layer turbulence, so the velocity scale is the friction velocity and the
    length scale is the vortex size, which grows linearly as the Jensen wake scale. The
    sqrt(24) converts a top-hat width to a Gaussian second moment. Every constant is
    fixed by the log law -- there is no fitted decay rate.

    Contrast Zong Sec 4.1's nu_E = 0.005*Gamma0, which is the Squire (1965) scaling on the
    vortex's OWN circulation and carries no ambient turbulence at all; it is 7-33x smaller
    over x/D = 2-10 (NREL-5MW, 25 deg, LES inflow; paper_metrics.py mNuRatioLo/Hi), which is why the model conserved circulation where the LES does not.
    """
    k = 0.4 / np.log(params.Zh / params.field_params.z0)
    u_star = k * params.Uh
    return u_star * 2.0 * k / np.sqrt(24.0)


def squire_nu(params, Uhub):
    """Zong Sec 4.1's core diffusivity, nu_E = 0.03*Gamma_0/(2pi) = 0.005*Gamma_0 -- the
    Squire (1965) scaling, in which the vortex is diffused by the turbulence it generates
    itself. Superseded by abl_nu_slope; retained only so the ablation in the paper can be
    reproduced without editing the model."""
    return 0.005 * abs(compute_gamma0(params, Uhub) * 0.45)


def vortex_adapter(params, local):
    """Builds a VortexSimConfig. field_params.vortex_nu selects the core-diffusion model:
    'abl' (default, Shapiro Eq 4.7) or 'squire' (the original PVT closure)."""
    if getattr(params.field_params, "vortex_nu", "abl") == "squire":
        nu_slope, nu_const = 0.0, squire_nu(params, local.Uhub)
    else:
        nu_slope, nu_const = abl_nu_slope(params), 0.0
    return VortexSimConfig(
        D=params.D,
        Uhub=local.Uhub,
        dgamma=jnp.asarray(compute_dgamma(params, local.Uhub)),
        calculation_domain=local.calculation_domain,
        phi=jnp.asarray(params.phi),
        beta=params.beta,
        Yoffset=params.Yoffset,
        Zhub=params.Zhub,
        nu_slope=nu_slope,
        nu_const=nu_const,
        Nv=params.Nv,
        V=jnp.asarray(local.V),
        W=jnp.asarray(local.W),
        yloc=jnp.asarray(params.yloc),
        zloc=jnp.asarray(params.zloc),
    )


def initial_vortex_state(params, local):
    """The fresh ring + hub + mirror vortices this rotor sheds, before any evolution."""
    from .vortex_model import _define_location
    return _define_location(vortex_adapter(params, local))[0]


def simulate_vortex_field(params, local, seed=None, total_steps=1000):
    """Evolve this rotor's vortex system. `seed` continues an existing cloud instead of
    starting a fresh ring -- used by the single-field march.
    """
    adapter = vortex_adapter(params, local)
    stacked, _was_active = _simulate_vortex_evolution_jit(
        adapter, params.field_params.merge_threshold * params.D,  # config value is in diameters
        params.field_params.cfl_factor, total_steps, seed,
    )
    return stacked


#: Which eddy-viscosity closure supplies the deficit's nu_T. Measured on all three datasets
#: with the conservative operator; the k_w budget is marched either way, so solver_ti_field
#: works in all three:
#:
#:                      Crespo      Du Eq (26)    k_w + sqrt(k)/S
#:   full-wake C/A/B    .953/.916/.920  .926/.895/.906  .908/.811/.836
#:   LES aligned/yawed  .931/.900       .919/.900       .892/.868
#:   8-turbine array    .954            .899            .845
#:   case C power T2/T3 +4.5/+4.7%      +45.1/-8.2%     +1.0/-41.5%
#:   case C bay bias    +8.4/-6.8/-7.6  +17.8/-11.0/-15.9  +4.9/-22.6/-25.6
#:
#: Crespo wins everywhere except bay-1 bias and T2-only power, and it wins for the right
#: reasons: per-bay R2 is uniformly > 0.95 with small mixed-sign bias, and depth/width/
#: integral are within ~13% at every station from 4D on -- not a cancellation signature.
#: The two k_w-based closures are both REGRESSIONS against it, which is why commit 5408bcc
#: ("replacing the Crespo correlation") is not the default. See [[bay-errors-compensate]].
#:
#: The open puzzle, and it is the interesting one: Crespo's TI is 12-33% TOO HIGH (case C,
#: bays 1/2/3) and gives the best velocity field, while the k_w closure's TI is accurate to
#: +-5% and gives the worst. So a correct TI fed through gradient diffusion UNDER-mixes the
#: time-averaged field. Candidate cause: the measured TI is a time standard deviation and
#: includes wake meandering, whose broadening of the MEAN wake is advective and cannot be
#: reproduced by nu_T grad^2 U from the same TI -- see [[les-wake-too-narrow-meandering]].
#: Caveat before treating the k_w TI as evidence: production scales with the deficit gradient
#: squared, so its too-sharp wake inflates its own k_w. That agreement is not independent.
NUT_CRESPO = False

#: Budget's nu_t from c sqrt(k) l_m instead of Du Eq (26), removing that correlation's three
#: fitted constants from the model. Inert when NUT_CRESPO is on.
BUDGET_PK = True

#: Multiplicative meandering factor on the DEFICIT's nu_T: nu_T = C_m * sqrt(c_mu) * k/S.
#:
#: The physical closure reproduces the measured TI to within a few per cent yet under-mixes
#: the TIME-AVERAGED field, while Crespo's nu_T -- fed a TI that is 12-33% too high, R2 = -6
#: as a TI predictor -- gives the best velocity field. DWM's architecture says why: a wake
#: "is dynamically meandering and of rather narrow nature. Only when looking at large time
#: averages, the wake appears to be static and rather broad", and superimposing meandering on
#: a steady wake "increased the width and reduced the intensity of the time-averaged velocity
#: distribution". All our validation data are time-averaged, so a steady gradient-diffusion
#: closure fed the true TKE must under-mix. C_m states that decomposition openly instead of
#: hiding it inside a correlation: k_w stays the physical TKE, C_m carries the meandering
#: contribution to time-averaged mixing.
#:
#: FALSIFIABILITY, and this is the point: if one C_m works across the wind tunnel, the LES and
#: the 8-turbine array it is a physical factor; if the optimum is dataset-dependent it is a
#: level knob and must be rejected on the same grounds as Crespo's m -- see
#: [[nut-crespo-m-error-cancellation]] and [[nut-calibration-unresolved]].
NUT_MEANDER_FACTOR = 1.0

#: Additive meandering diffusivity, utils.meander_diffusivity. Replaces the constant factor
#: above with the form the derivation and the LES snapshots both give: nu_m = U I_vf^2 x,
#: zero at the rotor and growing with travel distance. No fitted constant -- phi = 0.738 and
#: I_v/I_u = 0.735 are both measured. See [[meander-diffusivity-measured]].
MEANDER_NU = False

#: Age fed to the budget's l_m and Psi. Du's own reference is the STATION age, which
#: restarts at every rotor though T2's shear layer is already ~1D wide. The options were
#: measured against each other on all three datasets:
#:
#:   budget l_m        C/A/B full-wake      LES al/yaw     8-turb   X_STATION_TOL_D
#:   'station'         .9084/.8168/.8413    .8897/.8625    .8457        0.347%
#:   'farm'            .9542/.9107/.9127    .8524/.8272  DIVERGES         --
#:   'width'           .9317/.8449/.8622    .9101/.8734    .9109        0.347%
#:   (Crespo, ref)     .9530/.9155/.9199    .9308/.8999    .9535        9.325%
#:
#: 'farm' never restarts and wins the 3-turbine case, but l_m ~ x_farm never saturates: it
#: over-mixes at atmospheric scale and collapses the dt cap on 8 turbines even at 4x the step
#: budget. The two bracket the answer from opposite sides.
#:
#: 'width' is the literature form -- Klemmer & Howland (2025) JRSE 17, 033304 transport k and
#: set l_m to the wake width -- and it lands inside the bracket on every dataset while keeping
#: layout continuity exactly. It is implemented as a FLOOR on Du's age, at the width the
#: station was handed (adapter.w_in): zero at the first rotor, so its near wake is untouched
#: (bay-1 bias and T2 power are bit-unchanged from 'station'), and downstream it stops l_m
#: restarting below the scale already present. Measured, the wake width does not restart --
#: 0.978 at the end of bay 1, 1.049 at T2's rotor, 1.165, 1.196 at T3's, 1.290.
#:
#: w_in is measured over the rotor's own lateral footprint, NOT the whole cross-plane. A
#: whole-plane integral scores better (C .9410, 8-turbine .9315) but a bystander 7D to the
#: side then inflates it, pushing the X_STATION_TOL_D discrepancy to 1.92% -- reintroducing
#: the non-local coupling that disqualified Crespo. Half the gain for exact invariance is the
#: right trade here; see [[closure-owns-grouping-invariance]].
#: Distance in D at which the budget's length scale saturates. A TRANSPORTED length scale
#: grows early and then saturates on its own production/dissipation balance; capping the
#: farm-referenced age emulates that, and the cap is what makes the farm reference usable --
#: uncapped it over-mixes at atmospheric scale and collapses the dt cap on 8 turbines.
#:
#: Measured (cases C/A/B, LES aligned/yawed, 8-turbine array):
#:    cap    C      A      B      LES al  LES yaw  8-turb
#:    7.5  .9542  .9117  .9156   .9268   .8888    .9449
#:    10   .9544  .9092  .9145   .9226   .8804    .9479
#:    15   .9542  .9107  .9127   .8531  DIVERGED  DIVERGED
#:
#: 7.5 is a CALIBRATED constant, the one this closure has. Its admissible range is narrow --
#: 15 diverges, and 7.5 beats 10 on four of six cases -- but it was chosen from three values,
#: so a finer sweep is still owed. Physically it is the distance over which the turbulence
#: length scale reaches its farm-scale equilibrium.
#: What age newly produced turbulence carries in the transported-age field: True injects it
#: at age zero (a tracer age), False at the local age so dA/dx = 1 and A is the LAYER's
#: travel distance. See utils.advance_kwa.
AGE_PRODUCTION_RESETS = False

BUDGET_AGE_CAP = 7.5

#: Derive the saturation from ABL theory instead of the bare number above: the wake mixing
#: length cannot exceed the ABL's own, l_m <= kappa * z_h (Prandtl-Blackadar near-surface
#: limit). Inverting l_m = k_lm * x gives x_cap = kappa z_h / (k_lm D), which scales with hub
#: height and rotor diameter instead of being fitted in diameters.
#:
#: TESTED AND REJECTED, kept for the record. The fitted 7.5 D corresponds to
#: l_m,sat = 0.212 D = 26.7 m on the LES (kappa z_h = 36.0 m, ratio 0.74) and 0.195 D on the
#: tunnel (kappa z_h = 0.050 m, ratio 0.58) -- 0.6-0.75 of the ABL value across three orders
#: of magnitude, which is what motivated deriving it. Measured:
#:
#:                 C      A      B     LES al  LES yaw  8-turb
#:   cap 7.5 D   .9542  .9117  .9156   .9268   .8888    .9449
#:   kappa z_h   .9542  .9107  .9126   .9227   .8801   DIVERGED
#:
#: It diverges on the 8-turbine array, where it derives 14.13 D -- the same failure as the
#: 15 D sweep point. The deeper objection is the SCALING: if kappa z_h were the bound, the
#: optimal cap in diameters would differ per case (10.1 / 12.9 / 14.1 D for LES / full-wake /
#: array), but the measured preferences are 7.5 / 7.5 / ~10 D. The optimum tracks D, not
#: hub height, which is the opposite of what this hypothesis predicts.
#:
#: The D-scaling instead hints at a WAKE-scale bound: l_m,sat ~ 0.20 D against a saturated
#: wake width of ~1.29 D, i.e. the shear layer stops growing at ~0.15 of the wake it lives
#: in. That is an observation, not a derivation -- 0.15 is not the 0.08 plane-mixing-layer
#: value.
ABL_LENGTH_CAP = False

#: Age fed to the budget's l_m and Psi.
#:   'station'  Du's own reference; restarts at every rotor, under-mixes deep arrays by 25%
#:   'width'    floored at the incoming wake width; bounded, but grows too slowly to produce
#:              the T3 > T2 turn-around (T3/T2 = 0.92-0.95 against a measured 1.08)
#:   'farm'     from the farm's first rotor, saturating at BUDGET_AGE_CAP -- best accuracy,
#:              but DISQUALIFIED: it reads rotor positions, see below
#:   'transported'  the k_w-weighted age field, saturating at BUDGET_AGE_CAP -- the default
#:
#: 'transported' delivers 'farm' accuracy with 'station' locality. It solves the same equation
#: as k_w for M = k_w*A, with production injecting at the LOCAL age so dA/dx = 1 -- so A is the
#: distance the shear LAYER has travelled, accumulated through the field instead of read off
#: rotor positions. Measured side by side:
#:
#:                    C      A      B     LES al  LES yaw  8-turb  power  distant rotor
#:   transported    .9541  .9105  .9152   .9269   .8889   .9449   5.07%      0.17%
#:   farm cap 7.5   .9542  .9117  .9156   .9268   .8888   .9449   4.93%     12.26%
#:   width          .9317  .8449  .8622   .9101   .8734   .9109  13.4%       1.74%
#:   Crespo         .9530  .9155  .9199   .9308   .8999   .9535   6.3%          --
#:
#: i.e. within 0.005 mean R2 of Crespo, BETTER than Crespo on power, and at the numerical
#: floor on locality. A reaches 3.72 / 8.70 / 13.67 D at farm distances of 3.99 / 9.01 /
#: 14.01 D, so it tracks travel distance without ever being told a rotor position.
#:
#: The farm reference reproduces the deep-array turn-around too: the
#: real farm has T3 reading FASTER inflow than T2 (T3/T2 = 1.079 measured on case C) because
#: by the third row the turbulence has built up enough that bay 2 recovers faster than bay 1.
#: Measured T3/T2 = 1.045 / 1.066 / 1.117 on C/A/B against 1.079 / 1.043 / 1.169. No uniform
#: multiplier can do this -- NUT_MEANDER_FACTOR shifts both bays and leaves the ratio at 0.95.
#: The 5D row golden also now rises and plateaus (0.651, 0.673, 0.691, 0.681) instead of
#: declining monotonically -- see [[no-deep-array-plateau]].
#:
#: DISQUALIFIED FOR LAYOUT WORK, which is why 'width' is the default despite scoring worse.
#: Its X_STATION_TOL_D invariance is excellent (0.014%, better than the width floor's 0.347%,
#: because past saturation the exact x_farm stops mattering), but that test cannot see the
#: real defect. x_farm is the streamwise distance from the farm's FIRST rotor, irrespective of
#: lateral position, so a rotor placed off to the side AND upstream rewrites the length scale
#: for a column it never touches. Measured with A = [0, 7D], B = [5D, 0], probe = [10D, 0]:
#:
#:   budget l_m      same-x neighbour   distant UPSTREAM neighbour
#:   station               0.22%              0.17%
#:   width                 0.22%              1.74%
#:   farm cap 7.5          0.22%             12.26%
#:
#: The same-x column is the numerical floor and is identical for all three, so there is no
#: lateral leak -- the whole effect is x_farm reacting to which rotor happens to be first.
#: 12.26% is worse than the 9.325% that disqualified Crespo. See
#: test_distant_upstream_rotor_does_not_change_a_column, which the suite previously lacked:
#: test_lateral_neighbour_does_not_disturb_a_column puts its neighbour at the SAME x as the
#: first rotor, so x_farm is unchanged and the test is blind to this.
#:
#: The physics it captures is real (turbulence age matters, and eddies do not reset at a
#: rotor); it is the GLOBAL implementation that is wrong. Making it local needs the age
#: transported as a k_w-weighted field, not read off rotor positions.
BUDGET_AGE = 'transported'  

#: Exponent n in the turbulence-shortened near-wake length: the ramp is evaluated at
#: age * (TI_incident/TI_amb)^n, so a rotor placed in turbulent air completes its induction
#: development sooner. 0.0 reproduces Shapiro exactly and skips the branch entirely.
#:
#: Shapiro, Gayme & Meneveau (2018) set the development length from the rotor radius alone,
#: with no turbulence dependence, so the model holds ~1.1-1.2D at every rotor. Measured on the
#: full-wake set the wake minimum sits 1.7D behind T1 but only 0.4D behind T2 and T3 -- the
#: near wake SHORTENS down the row. Physical basis: it ends where the shear layer, growing at
#: a rate proportional to the turbulence, reaches the axis, so x_0 ~ D/I (Vermeulen 1980;
#: Bastankhah & Porte-Agel 2016) -- n = 1 is that scaling.
#:
#: Measured with the Crespo closure, case C bay 2, and provably inert at T1 (bay-1 minimum,
#: bay-1 bias and T2 power are unchanged at every n, because ti_in is the INCIDENT value):
#:
#:    n    min offset   min depth (meas 0.377)   recovery (meas +0.0699)   C/A/B R2
#:    0      +0.7D         0.365  (-3%)            +0.0966  (+38%)        .9530/.9155/.9199
#:    1      +0.3D         0.320 (-15%)            +0.0983  (+41%)        .9564/.9182/.9236
#:    2      +0.0D         0.299 (-21%)            +0.0979  (+40%)        .9566/.9179/.9238
#:
#: The length error is real and this fixes it, but it is NOT the dominant defect, and the
#: test refuted the hypothesis it was built on: the +38% over-fast recovery is completely
#: unmoved, so the late minimum and the over-recovery are INDEPENDENT, not a compensating
#: pair. Shortening the ramp also injects the full deficit sooner, so the minimum deepens as
#: it moves -- depth and length are coupled here and cannot both be fixed from the ramp.
#: Left at 0.0 so the next diagnosis is not confounded; n = 1 is the only defensible nonzero
#: value (n = 2 fits the location better but has no physical basis and costs 21% in depth).
#: See [[near-wake-length-fixed]] and [[no-deep-array-plateau]].
RAMP_TI_EXP = 0.0 

#: ON by default. It is less accurate than Crespo on every validation case (see above), and
#: it is the only closure that satisfies the layout-continuity invariants an optimiser needs:
#:
#:                                     Crespo    this closure
#:   X_STATION_TOL_D invariance         9.325%      0.347%
#:   lateral |dy| = 3D gate             0.327%      0.400%   (benign in both)
#:
#: Crespo must be TOLD which rotors are upstream, so its nu_T depends on how the march was
#: cut; 9.3% is the step a layout optimiser sees when two rotors cross a 0.25D streamwise
#: offset, and BO will exploit it. The stated goal of this repo is layout and yaw
#: optimisation on oblique and irregular layouts, so continuity is the binding requirement
#: and validation R2 is not. See [[closure-owns-grouping-invariance]].
#:
#: The accuracy gap is understood and has a known route: it is the deep-array under-recovery,
#: owned by the budget's restarting length scale, and a non-restarting one recovers
#: Crespo-level accuracy (case C 0.9542 vs 0.9530) -- but unbounded it diverges on 8 turbines,
#: so the fix is a TRANSPORTED length scale. See [[budget-length-scale-brackets]].
NUT_FROM_TKE = True

#: Distance in D past which the erf ramp has saturated, so a carried rotor's term is
#: mathematically zero and its slot can be left empty. Bounds the PHYSICS, not the bits:
#: carrying such a rotor still reassociates the tensordot below, which moved the 5D row by
#: 1.7e-13. Irrelevant physically, but do not claim bit-identity from this constant.
RAMP_REACH_D = 6.0


def axial_induction_ramped(Ct_eff, x_D):
    """Axial induction with the near-wake pressure-gradient development of Shapiro,
    Gayme & Meneveau (2018), as used by Zong & Porte-Agel (2020) JFM 889 A8 Sec 3.1:

        Ct(x) = Ct * (1 + erf(x/D)) / 2,   a(x) = (1 - sqrt(1 - Ct(x)/cos(beta))) / 2

    The deficit grows from a(0) at the rotor to the fully-developed value by x ~ 2D
    instead of appearing all at once. Zong's own transport equation cannot create
    deficit downstream -- it only advects and diffuses -- so injecting the far-wake
    value at x=0 leaves the model correct at 0.5D and ~17% shallow by 2D, where the
    measured wake is still deepening.
    """
    ramp = 0.5 * (1.0 + jerf(x_D))
    return 0.5 * (1.0 - jnp.sqrt(jnp.maximum(1.0 - Ct_eff * ramp, 0.0)))


def _ct_eff(params):
    return float(params.Ct_yawed / np.cos(params.beta))


#: Members of a stacked VortexField that carry NO leading frame axis. The cross-plane
#: grid is identical at every step, so the marches store one copy instead of one per
#: step; see vortex_model._simulate_vortex_evolution_jit.
_UNSTACKED_MEMBERS = ("yloc", "zloc")


def _frame(stacked, i, unstacked=_UNSTACKED_MEMBERS):
    """Frame `i` of a stacked VortexField, leaving the unstacked members alone.

    Replaces a plain tree_map(leaf[i]), which would slice the grid's first row instead
    of selecting a frame. `unstacked` differs between the two marches: U carries no
    frame axis coming out of the vortex march (initialize_wake_field stores the single
    seed field), but does carry one coming out of the deficit march, where it is the
    quantity being marched.
    """
    sliced = {f.name: getattr(stacked, f.name)[i]
              for f in dataclasses.fields(stacked)
              if f.name not in unstacked}
    return dataclasses.replace(stacked, **sliced)


def _disc_mask(yloc, zloc, p, dy):
    """Rotor p's disc on a grid whose local y is offset from p's own by `dy`.

    Every turbine shares one absolute cross-plane grid, but `yloc` is stored local to
    each rotor, so a companion rotor at the same station sits at yloc + dy where
    dy = lead.pos[1] - p.pos[1].
    """
    r2 = ((yloc + dy + p.Yoffset) ** 2) / (np.cos(p.beta) ** 2) + (zloc - p.Zhub) ** 2
    return np.sqrt(r2) <= p.R


def initialize_wake_field(params, stacked, local, companions=(), kw0=None, kwa0=None):
    yloc = np.asarray(stacked.yloc)
    zloc = np.asarray(stacked.zloc)
    beta = params.beta

    dl = params.dl  # equivalent to yloc[1,0]-yloc[0,0]; verified in Step 1 sub-step 5
    Uin = np.asarray(local.Uin)
    U = Uin.copy()
    # Rotors sharing this station shed simultaneously, so all their discs are carved
    # into the same initial condition. `companions` is empty for one turbine per x,
    # which is the single-row case and reduces to the original single-disc carve.
    for p, dy in ((params, 0.0),) + tuple(companions):
        mask = _disc_mask(yloc, zloc, p, dy)
        a0 = float(0.5 * (1.0 - np.sqrt(max(1.0 - _ct_eff(p) * 0.5, 0.0))))  # erf(0) = 0
        U[mask] -= 2.0 * U[mask] * a0

    U_smooth = Uin - smooth_2d(Uin - U, kernel_size=3)

    kw_seed = (np.zeros_like(U_smooth) if kw0 is None
               else np.asarray(kw0).reshape(U_smooth.shape))
    kwa_seed = (np.zeros_like(U_smooth) if kwa0 is None
                else np.asarray(kwa0).reshape(U_smooth.shape))
    new_stacked = dataclasses.replace(
        stacked,
        U=jnp.asarray(U_smooth),
        kw=jnp.asarray(kw_seed),
        kwa=jnp.asarray(kwa_seed),
        X=stacked.X.at[0].set(0.0),
        t=stacked.t.at[0].set(0.0),
    )
    return new_stacked, dl


def calculate_deficit_field(params, local, stacked, dl, upstream_turbines, N_upstream_max=None,
                            max_steps=1500, companions=(), carried=(), n_slots=None,
                            w_in=0.0):
    if N_upstream_max is None:
        N_upstream_max = len(upstream_turbines)
    up_a, up_D, up_pos_x, up_pos_y, up_Uhub, up_mask = pack_upstream_turbines(
        upstream_turbines, N_upstream_max)

    yloc = np.asarray(params.yloc)
    zloc = np.asarray(params.zloc)
    # (params, dy, x_off). x_off is how far UPSTREAM of this station the rotor sits, so
    # its near wake is x_off older than the station-local march distance. Station members
    # are at x_off = 0 and reproduce the original single-origin behaviour exactly.
    rotors = (((params, 0.0, 0.0),)
              + tuple((p, dy, 0.0) for p, dy in companions)
              + tuple(carried))
    # Fixed stack length, as pack_upstream_turbines does for the upstream arrays: the
    # stack grows by one station per march, so a length that tracked it would retrace the
    # scan once per station. Padding slots keep Ct_eff = 0, which makes a_old = a_new = 0
    # and the ramp factor exactly 1.0 in any dtype -- see
    # test_completed_and_padded_rotors_are_inert_in_the_ramp_stack.
    n_slots = max(len(rotors) if n_slots is None else n_slots, len(rotors))
    rotor_mask = np.zeros((n_slots,) + yloc.shape)
    Ct_eff = np.zeros(n_slots)
    x_off = np.zeros(n_slots)
    for i, (p, dy, off) in enumerate(rotors):
        # Beyond RAMP_REACH_D the erf has saturated and the factor is 1 to within 1e-12,
        # so the slot is left inert rather than paying for a mask that cannot change U.
        # This is an optimisation only; including it would give the same answer.
        if off >= RAMP_REACH_D * p.D:
            continue
        rotor_mask[i] = np.asarray(smooth_2d(_disc_mask(yloc, zloc, p, dy).astype(float),
                                             kernel_size=3))
        Ct_eff[i] = _ct_eff(p)
        x_off[i] = off

    # Incident turbulence at each rotor, from the k_w the station was HANDED -- not from the
    # evolving field. Keying the ramp to current.kw instead let T1 shorten its own near wake
    # with turbulence its own wake had produced: measured, bay-1 bias went +8.4% -> +12.4%
    # and the minimum moved 0.2D ahead of the measurement.
    kw_seed = np.asarray(stacked.kw)
    if kw_seed.ndim == 3:
        kw_seed = kw_seed[0]
    U0_ = float(nominal_hub_velocity(params))
    k_B_ = 1.5 * ((params.field_params.I_amb / 1.28) * U0_) ** 2
    ti_in = np.full(n_slots, (params.field_params.I_amb / 1.28))
    if kw_seed.size:
        for i in range(n_slots):
            w = rotor_mask[i]
            if w.sum() > 0:
                ti_in[i] = np.sqrt(max(2.0 / 3.0 * (k_B_ + float((kw_seed * w).sum() / w.sum())),
                                       0.0)) / U0_

    adapter = DeficitFieldConfig(
        pos=jnp.asarray(params.pos), D=params.D, Uhub=local.Uhub, Uin=jnp.asarray(local.Uin), Zhub=params.Zhub,
        U0=float(nominal_hub_velocity(params)),
        rotor_mask=jnp.asarray(rotor_mask), Ct_eff=jnp.asarray(Ct_eff),
        x_off=jnp.asarray(x_off),
        x_farm=float(max((off for _, _, off in carried), default=0.0)),
        ti_in=jnp.asarray(ti_in),
        w_in=float(w_in),
    )
    dt_cap = min(dl / local.Uhub, 0.25 * params.D / local.Uhub)  # constant for the whole run -- plain floats

    seed = _frame(stacked, 0, unstacked=_UNSTACKED_MEMBERS + ("U", "kw", "kwa"))
    stacked_out, was_active = _calculate_deficit_field_jit(
        stacked, seed, adapter, params.field_params.I_amb, params.field_params.WV,
        up_a, up_D, up_pos_x, up_pos_y, up_Uhub, up_mask, dl, dt_cap,
        local.calculation_domain, max_steps,
    )
    # was_active[k] corresponds to buffer index k+1 (the loop below is 1-indexed exactly like
    # the original); kept_count includes the seed (index 0) plus every real advance.
    kept_count = int(jnp.sum(was_active)) + 1

    grid = (np.asarray(seed.yloc), np.asarray(seed.zloc))
    frames = [_vortex_field_to_numpy(seed, grid)]
    frames += [_vortex_field_to_numpy(_frame(stacked_out, i), grid)
               for i in range(kept_count - 1)]
    return frames


@partial(jax.jit, static_argnames=("max_steps",))
def _calculate_deficit_field_jit(stacked, seed, adapter, I_amb, WV, up_a, up_D, up_pos_x,
                                  up_pos_y, up_Uhub, up_mask,
                                  dl, dt_cap, calculation_domain, max_steps):
    def scan_step(carry, _):
        current, active = carry

        def when_active(_):
            # Wake-added TKE replaces the Crespo correlation. The budget's own nu_t and
            # Psi take the AMBIENT turbulence intensity, as Du intends -- that keeps the
            # march non-circular, since the deficit's nu_T below then uses the local TI
            # that k_w produces. age = current.X/D is the distance from THIS station's
            # rotor, i.e. the shear-layer age, which is the reference that works: taking
            # it from the farm inlet lets Psi grow without bound down a row.
            age_D = jnp.maximum(current.X / adapter.D, 1e-6)
            ti_amb = I_amb / 1.28
            k_B_amb = ambient_tke(I_amb, adapter.U0)
            if BUDGET_AGE == 'station':
                bud_age = age_D
            elif BUDGET_AGE == 'farm':
                # Capped: a TRANSPORTED length scale grows early and then saturates, so
                # capping the farm age emulates one and brackets what it could buy. Uncapped
                # it diverges on 8 turbines; see BUDGET_AGE_CAP.
                cap = (KARMAN * adapter.Zhub / ((0.2 * ti_amb + 0.015) * adapter.D)
                       if ABL_LENGTH_CAP else BUDGET_AGE_CAP)
                bud_age = jnp.minimum(age_D + adapter.x_farm / adapter.D, cap)
            elif BUDGET_AGE == 'transported':
                # LOCAL age, from the k_w-weighted field. Floored at the station-local age so
                # the fresh near wake keeps Du's short-age branch, where k_w is still tiny and
                # A is numerically meaningless.
                cap = (KARMAN * adapter.Zhub / ((0.2 * ti_amb + 0.015) * adapter.D)
                       if ABL_LENGTH_CAP else BUDGET_AGE_CAP)
                bud_age = jnp.maximum(
                    age_from_kwa(current.kwa, current.kw, adapter.D, 0.0, cap), age_D)
            else:                                   # 'width'
                # Du's age governs the fresh shear layer, FLOORED at the scale the station
                # inherited. The floor is zero at the first rotor, so its near wake is
                # untouched; downstream it stops l_m restarting below the incoming scale.
                floor = LM_OVER_WIDTH * (adapter.w_in / adapter.D) / (0.2 * ti_amb + 0.015)
                bud_age = jnp.maximum(age_D, floor)
            nut_bud = (nu_t_budget(current.kw, I_amb, bud_age, adapter.U0, adapter.D)
                       if BUDGET_PK else
                       du_nu_t(ti_amb, bud_age, adapter.U0, adapter.D))
            psi_bud = du_psi(ti_amb, bud_age, adapter.D)
            if NUT_CRESPO:
                NuT = NuT_model(current, adapter, I_amb, up_a, up_D, up_pos_x,
                                up_pos_y, up_Uhub, up_mask)
            elif NUT_FROM_TKE:
                NuT = NUT_MEANDER_FACTOR * nu_t_from_tke(
                    current.kw, I_amb, adapter.U0,
                    adapter.Uin - current.U, current.yloc, current.zloc)
            else:
                NuT = du_nu_t(ti_from_kw(current.kw, I_amb, adapter.U0), age_D,
                              adapter.U0, adapter.D)
            if MEANDER_NU:
                NuT = NuT + meander_diffusivity(I_amb, age_D, adapter.U0, adapter.D)
            # NuT is a FIELD now, but the step must stay scalar, so the diffusive limit
            # is taken at its largest value on the plane. For a uniform NuT this is the
            # same number it always was.
            dt = jnp.minimum(dt_cap, (dl ** 2) / (2 * (jnp.max(NuT) + 1e-6)))

            new = interpolate_vec_data(stacked, current.t + dt)
            U, X_new = advance_wake_field(current, dt, NuT, adapter, WV)
            Uw_bud = adapter.Uin - current.U
            dWy, dWz = jnp.gradient(Uw_bud, current.yloc[:, 0], current.zloc[0, :])
            prod_bud = nut_bud * (dWy ** 2 + dWz ** 2)
            kwa_new = advance_kwa(current.kwa, current.kw, current.yloc, current.zloc,
                                  X_new - current.X, nut_bud, psi_bud, adapter.U0,
                                  prod=None if AGE_PRODUCTION_RESETS else prod_bud)
            kw_new = advance_kw(current.kw, current.U, adapter.Uin, current.yloc,
                                current.zloc, X_new - current.X, nut_bud, psi_bud,
                                adapter.U0)

            # EACH ROTOR AGES ON ITS OWN STREAMWISE ORIGIN.
            #
            # The factor is a ratio of consecutive ages, so over a march it telescopes to
            # (1 - 2a(age_end)) / (1 - 2a(age_start)), and the initial condition is carved
            # with exactly a(0) -- a full segment therefore composes to the correctly
            # developed deficit. But the stack used to hold only the current station's
            # rotors and age them all by current.X, so a rotor dropped out when the march
            # advanced and its ramp froze wherever it had reached. Cut a segment to
            # L << 2D and the deficit sticks near the half-strength carve value 1 - 2a(0):
            # a 3x3 farm rotated 0.01 deg, geometrically indistinguishable from aligned,
            # read 21% high once its rows split into separate stations.
            #
            # Adding x_off and keeping shed rotors in the stack makes the handover exact,
            # not merely better. A rotor leaves segment p at age dx_station + x_off_p
            # (the field is read at the interpolation distance, not at need_m) and
            # re-enters at x_off_{p+1} = x_off_p + dx_station -- the same number, so the
            # telescoping continues across the cut with no jump.
            age_old = (current.X + adapter.x_off) / adapter.D
            age_new = (X_new + adapter.x_off) / adapter.D
            if RAMP_TI_EXP:
                # Incident TI per rotor (adapter.ti_in), i.e. the turbulence the rotor is
                # placed INTO. A rotor deep in the farm sees what its predecessors left --
                # exactly the quantity Shapiro's radius-only length scale omits.
                shorten = (jnp.maximum(adapter.ti_in, ti_amb) / ti_amb) ** RAMP_TI_EXP
                age_old = age_old * shorten
                age_new = age_new * shorten
            a_old = axial_induction_ramped(adapter.Ct_eff, age_old)
            a_new = axial_induction_ramped(adapter.Ct_eff, age_new)
            ramp = (1.0 - 2.0 * a_new) / jnp.maximum(1.0 - 2.0 * a_old, 1e-6)
            U = U * (1.0 + jnp.tensordot(ramp - 1.0, adapter.rotor_mask, axes=1))

            new = dataclasses.replace(new, U=U, kw=kw_new, kwa=kwa_new, X=X_new,
                                      t=current.t + dt)

            still_active = new.X <= calculation_domain  # this crossing entry stays the final real one
            return new, still_active

        def when_inactive(_):
            return current, active

        next_state, next_active = lax.cond(active, when_active, when_inactive, operand=None)
        emitted = dataclasses.replace(next_state, yloc=_UNSTACKED, zloc=_UNSTACKED)
        return (next_state, next_active), (emitted, active)

    init_carry = (seed, True)
    _, (stacked_out, was_active) = lax.scan(scan_step, init_carry, xs=None, length=max_steps)
    stacked_out = dataclasses.replace(stacked_out, yloc=seed.yloc, zloc=seed.zloc)
    return stacked_out, was_active


def _vortex_field_to_numpy(vortex_field, grid=None):
    """Converts every array field of a VortexField from jnp back to plain numpy, and t/X
    (0-d jnp arrays once produced inside jax-based code) back to Python floats -- the Loop-2/
    downstream (save_results, plotting) boundary contract.

    `grid` is an optional pre-converted (yloc, zloc) pair. Every frame of a march has the
    same cross-plane grid, so converting it per frame would give each of the hundreds of
    retained frames its own identical copy; pass it once and share the reference. Nothing
    downstream writes to yloc/zloc.
    """
    yloc, zloc = grid if grid is not None else (np.asarray(vortex_field.yloc),
                                                np.asarray(vortex_field.zloc))
    return dataclasses.replace(
        vortex_field,
        Y=np.asarray(vortex_field.Y), Z=np.asarray(vortex_field.Z),
        Rv=np.asarray(vortex_field.Rv), Circ=np.asarray(vortex_field.Circ),
        age=np.asarray(vortex_field.age),
        active=np.asarray(vortex_field.active),
        yloc=yloc, zloc=zloc,
        V=np.asarray(vortex_field.V), W=np.asarray(vortex_field.W),
        U=np.asarray(vortex_field.U), OmegaX=np.asarray(vortex_field.OmegaX),
        kw=np.asarray(vortex_field.kw),
        kwa=np.asarray(vortex_field.kwa),
        t=float(vortex_field.t), X=float(vortex_field.X),
    )
