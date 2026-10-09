import dataclasses
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import erf as jerf

from .turbine_state import LocalConditions, VortexSimConfig, DeficitFieldConfig, pack_upstream_turbines
from .utils import smooth_2d, NuT_model
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


def compute_dgamma(params, Uhub):
    """Shed circulation per azimuthal element, with the torque- and force-driven parts
    scaled separately.
    """
    gamma_ref = -compute_gamma0(params, Uhub) * 0.45  # Zong Eq (4.3), his own k

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


def initialize_wake_field(params, stacked, local, companions=()):
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

    new_stacked = dataclasses.replace(
        stacked,
        U=jnp.asarray(U_smooth),
        X=stacked.X.at[0].set(0.0),
        t=stacked.t.at[0].set(0.0),
    )
    return new_stacked, dl


def calculate_deficit_field(params, local, stacked, dl, upstream_turbines, N_upstream_max=None,
                            max_steps=1500, companions=()):
    if N_upstream_max is None:
        N_upstream_max = len(upstream_turbines)
    up_a, up_D, up_pos_x, up_Uhub, up_mask = pack_upstream_turbines(upstream_turbines, N_upstream_max)

    yloc = np.asarray(params.yloc)
    zloc = np.asarray(params.zloc)
    rotors = ((params, 0.0),) + tuple(companions)
    rotor_mask = np.stack([np.asarray(smooth_2d(_disc_mask(yloc, zloc, p, dy).astype(float),
                                                kernel_size=3))
                           for p, dy in rotors])
    Ct_eff = np.array([_ct_eff(p) for p, _ in rotors])

    adapter = DeficitFieldConfig(
        pos=jnp.asarray(params.pos), D=params.D, Uhub=local.Uhub, Uin=jnp.asarray(local.Uin), Zhub=params.Zhub,
        U0=float(nominal_hub_velocity(params)),
        rotor_mask=jnp.asarray(rotor_mask), Ct_eff=jnp.asarray(Ct_eff),
    )
    dt_cap = min(dl / local.Uhub, 0.25 * params.D / local.Uhub)  # constant for the whole run -- plain floats

    seed = _frame(stacked, 0, unstacked=_UNSTACKED_MEMBERS + ("U",))
    stacked_out, was_active = _calculate_deficit_field_jit(
        stacked, seed, adapter, params.field_params.I_amb, params.field_params.WV,
        up_a, up_D, up_pos_x, up_Uhub, up_mask, dl, dt_cap, local.calculation_domain, max_steps,
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
def _calculate_deficit_field_jit(stacked, seed, adapter, I_amb, WV, up_a, up_D, up_pos_x, up_Uhub, up_mask,
                                  dl, dt_cap, calculation_domain, max_steps):
    def scan_step(carry, _):
        current, active = carry

        def when_active(_):
            NuT = NuT_model(current, adapter, I_amb, up_a, up_D, up_pos_x, up_Uhub, up_mask)
            dt = jnp.minimum(dt_cap, (dl ** 2) / (2 * (NuT + 1e-6)))  # stability condition

            new = interpolate_vec_data(stacked, current.t + dt)
            U, X_new = advance_wake_field(current, dt, NuT, adapter, WV)

            # One station may carry several rotors side by side. They share an
            # x-origin, so current.X is common and only Ct_eff and the disc differ;
            # the masks do not overlap, so the per-rotor ramps simply add.
            a_old = axial_induction_ramped(adapter.Ct_eff, current.X / adapter.D)
            a_new = axial_induction_ramped(adapter.Ct_eff, X_new / adapter.D)
            ramp = (1.0 - 2.0 * a_new) / jnp.maximum(1.0 - 2.0 * a_old, 1e-6)
            U = U * (1.0 + jnp.tensordot(ramp - 1.0, adapter.rotor_mask, axes=1))

            new = dataclasses.replace(new, U=U, X=X_new, t=current.t + dt)

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
        t=float(vortex_field.t), X=float(vortex_field.X),
    )
