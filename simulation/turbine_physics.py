import dataclasses
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import erf as jerf

from .turbine_state import LocalConditions, VortexSimConfig, DeficitFieldConfig
from .utils import (smooth_2d, du_nu_t, advance_kw, velocity_floor, du_eps_coef,
                    advect_incident_ti, IU_TO_TI)
from .vortex_model import _simulate_vortex_evolution_jit, _UNSTACKED, march_until_stopped
from .model_solver import advance_wake_field, representative_velocity
from .field_ops import interpolate_vec_data


def inflow_profile(field_params, Uh, Zh, Z):
    """Undisturbed streamwise velocity at height Z: logarithmic, or uniform at Uh."""
    if str(getattr(field_params, "inflow", "log")).lower() == "uniform":
        return np.full_like(np.asarray(Z, dtype=float), float(Uh))
    z0 = field_params.z0
    zsafe = np.maximum(Z, z0 + 1e-6)
    return Uh * (np.log(zsafe / z0) / np.log(Zh / z0))


def init_Uin(params):
    return inflow_profile(params.field_params, params.Uh, params.Zh, params.zloc)


def rotor_disc_mask(params):
    """The yaw-projected rotor disc as a BOOLEAN, for diagnostics that want cell counts."""
    r2 = (((params.yloc + params.Yoffset) ** 2) / (np.cos(params.beta) ** 2)
          + (params.zloc - params.Zhub) ** 2)
    return np.sqrt(r2) <= params.R


def disc_weights_on(yloc, zloc, p, dy):
    """Fractional cell coverage of rotor p's disc on a grid whose local y is offset by dy."""
    y = np.asarray(yloc)
    z = np.asarray(zloc)
    dl_cell = np.sqrt((y[1, 0] - y[0, 0]) * (z[0, 1] - z[0, 0]))
    return _disc_fraction(y, z, p, dy, dl_cell)


def rotor_disc_weights(params):
    """disc_weights_on for a rotor reading its own grid."""
    return disc_weights_on(params.yloc, params.zloc, params, 0.0)


def rotor_average(U, w):
    """Coverage-weighted arithmetic disc mean -- the speed the AERODYNAMICS use."""
    u = np.asarray(U)
    w = np.asarray(w, dtype=float)
    return float(np.sum(w * u) / np.sum(w))


def rotor_equivalent_cubic(U, w):
    """Cubic rotor-equivalent wind speed, (<U^3>)^(1/3) over the disc."""
    u = np.asarray(U)
    w = np.asarray(w, dtype=float)
    return float((np.sum(w * u ** 3) / np.sum(w)) ** (1.0 / 3.0))


def nominal_hub_velocity(params):
    """Rotor-averaged undisturbed inflow -- depends only on config, never on the mutated Uin."""
    return rotor_average(init_Uin(params), rotor_disc_weights(params))


def nominal_hub_velocity_cubic(params):
    """nominal_hub_velocity's cubic counterpart, the reference for the power ratio."""
    return rotor_equivalent_cubic(init_Uin(params), rotor_disc_weights(params))


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
    """Total circulation shed from the blade tips, Zong & Porte-Agel (2020) Eq (4.3): Gamma0 =..."""
    omega = compute_omega(params, Uhub)
    return np.pi * Uhub ** 2 * params.Ct_yawed / omega


def compute_Ut(params, Uhub, beta=None, a=None):
    """Blade-tip velocity around the azimuth."""
    beta = params.beta if beta is None else beta
    omega = compute_omega(params, Uhub)
    z = params.Zhub + params.D / 2.0 * np.sin(params.phi)
    Utbl = inflow_profile(params.field_params, params.Uh, params.Zh, z)
    cb = np.cos(beta)
    a = params.a_at(beta) if a is None else a
    U_tx = (1 - a) * Utbl - omega * params.R * np.sin(params.phi) * np.sin(beta)
    U_ty = -omega * params.R * np.sin(params.phi) * cb
    U_tz = omega * params.R * np.cos(params.phi)
    return np.array([U_tx, U_ty, U_tz]).T


def _shed_shape(params, Uhub, beta, a=None):
    """Unnormalised azimuthal distribution of shed circulation, Zong Eq (4.4)."""
    Ut = compute_Ut(params, Uhub, beta, a)
    alpha = np.arcsin(Ut[:, 0] / np.sqrt(np.sum(Ut ** 2, axis=1)))
    return np.sin(alpha) * params.dphi


GAMMA_TRAILING_K = 0.45


def compute_dgamma(params, Uhub):
    """Shed circulation per azimuthal element, with the torque- and force-driven parts scaled..."""
    # The spin direction enters here and nowhere else: asym_yaw below is rescaled to M_target,
    # whose sign comes from the yaw, so the yaw-driven lateral impulse is left untouched.
    rot = float(np.ravel(getattr(params.config, "rotation", 1.0))[0])
    gamma_ref = -compute_gamma0(params, Uhub) * GAMMA_TRAILING_K * rot

    shape_b = _shed_shape(params, Uhub, params.beta_aero, params.a)
    shape_0 = _shed_shape(params, Uhub, 0.0)
    dgamma_b = gamma_ref / np.sum(shape_b) * shape_b
    dgamma_0 = gamma_ref / np.sum(shape_0) * shape_0

    asym_yaw = (dgamma_b - dgamma_b.mean()) - (dgamma_0 - dgamma_0.mean())
    z_rel = params.R * np.sin(params.phi)
    M_yaw = float(np.sum(asym_yaw * z_rel))
    M_target = (0.5 * Uhub * (np.pi * params.D ** 2 / 4.0) * params.Ct
                * np.sin(params.beta_aero) * np.cos(params.beta_aero) ** 2)
    if abs(M_yaw) > 1e-12:
        asym_yaw = asym_yaw * (M_target / M_yaw)
    # An unyawed rotor carries no lateral force, so the first azimuthal moment a sheared inflow
    # puts in the baseline shape is removed here (it sums to zero round the ring, so the
    # torque-derived total is unchanged); fixing the yaw part instead would diverge as beta -> 0.
    dgamma_0 = dgamma_0 - z_rel * (float(np.sum(dgamma_0 * z_rel)) / float(np.sum(z_rel ** 2)))
    return dgamma_0 + asym_yaw


def power_and_reference(params, Uhub3):
    """One rotor's power and its undisturbed reference, up to the common pi/4 * rho."""
    A = params.D ** 2
    P = A * params.Cp * (Uhub3 ** 3)
    P0 = A * params.config.Cp * (nominal_hub_velocity_cubic(params) ** 3)
    return P, P0


def calculate_efficiency(params, Uhub3):
    """Power ratio against the same rotor in undisturbed inflow."""
    P, nominal_P = power_and_reference(params, Uhub3)
    return P / nominal_P


def abl_nu_slope(params):
    """d(nu_T)/dx for a vortex in the ABL -- Shapiro, Gayme & Meneveau (JFM), "Generation and decay of..."""
    k = 0.4 / np.log(params.Zh / params.field_params.z0)
    u_star = getattr(params.field_params, "u_star", None)
    u_star = k * params.Uh if u_star is None else float(u_star)
    return u_star * 2.0 * k / np.sqrt(24.0)


def squire_nu(params, Uhub):
    """Zong Sec 4.1's core diffusivity, nu_E = 0.03*Gamma_0/(2pi) = 0.005*Gamma_0 -- the Squire (1965)..."""
    return 0.005 * abs(compute_gamma0(params, Uhub) * GAMMA_TRAILING_K)


_PM_CACHE = {}


def _pm_config(params):
    """The PM kernels for this rotor's grid, or None to use the exact pairwise sum."""
    fp = params.field_params
    nmin = getattr(fp, "pm_min_rotors", None)
    if nmin is None or getattr(params, "n_rotors", 1) < nmin:
        return None, 0.0, 0.0
    from .vortex_model import pm_ladder, pm_kernels
    tol = float(getattr(fp, "pm_tol", 0.05))
    cores, lo, lnr = pm_ladder(params.D, tol)
    key = (params.yloc.shape, float(params.yloc[1, 0] - params.yloc[0, 0]),
           float(params.zloc[0, 1] - params.zloc[0, 0]), float(params.zloc[0, 0]),
           round(lo, 12), round(lnr, 12), len(cores))
    if key not in _PM_CACHE:
        _PM_CACHE[key] = pm_kernels(cores, params.yloc, params.zloc)
    Av, Aw = _PM_CACHE[key]
    return (Av, Aw), lo, lnr


def vortex_adapter(params, local):
    """Builds a VortexSimConfig."""
    if getattr(params.field_params, "vortex_nu", "abl") == "squire":
        nu_slope, nu_const = 0.0, squire_nu(params, local.Uhub)
    else:
        nu_slope, nu_const = abl_nu_slope(params), 0.0
    _pm, _pm_lo, _pm_lnr = _pm_config(params)
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
        U_conv=float(local.Uhub * (1.0 - params.a)),
        Nv=params.Nv,
        V=jnp.asarray(local.V),
        W=jnp.asarray(local.W),
        yloc=jnp.asarray(params.yloc),
        zloc=jnp.asarray(params.zloc),
        pm_Av=None if _pm is None else _pm[0],
        pm_Aw=None if _pm is None else _pm[1],
        pm_lo=_pm_lo,
        pm_lnr=_pm_lnr,
    )


def initial_vortex_state(params, local):
    """The fresh ring + hub + mirror vortices this rotor sheds, before any evolution."""
    from .vortex_model import _define_location
    return _define_location(vortex_adapter(params, local))[0]


def simulate_vortex_field(params, local, seed=None, total_steps=1000):
    """Evolve this rotor's vortex system."""
    adapter = vortex_adapter(params, local)
    stacked, was_active = _simulate_vortex_evolution_jit(
        adapter, params.field_params.merge_threshold * params.D,
        params.field_params.cfl_factor, total_steps, seed,
    )
    return stacked, bool(was_active[-1])


RAMP_REACH_D = 6.0

DIFFUSIVE_CFL = 0.9

NACELLE_CD = 1.0

AGE_BLEND_EPS = 0.01


def axial_induction_ramped(Ct_eff, x_D):
    """Axial induction with the near-wake pressure-gradient development of Shapiro, Gayme & Meneveau..."""
    ramp = 0.5 * (1.0 + jerf(x_D))
    return 0.5 * (1.0 - jnp.sqrt(jnp.maximum(1.0 - Ct_eff * ramp, 0.0)))


def _ct_eff(params):
    return float(params.Ct_yawed / np.cos(params.beta))


_UNSTACKED_MEMBERS = ("yloc", "zloc")


def _frame(stacked, i, unstacked=_UNSTACKED_MEMBERS):
    """Frame `i` of a stacked VortexField, leaving the unstacked members alone."""
    sliced = {f.name: getattr(stacked, f.name)[i]
              for f in dataclasses.fields(stacked)
              if f.name not in unstacked}
    return dataclasses.replace(stacked, **sliced)


def _disc_mask(yloc, zloc, p, dy, radius=None):
    """Rotor p's disc on a grid whose local y is offset from p's own by `dy`."""
    r2 = ((yloc + dy + p.Yoffset) ** 2) / (np.cos(p.beta) ** 2) + (zloc - p.Zhub) ** 2
    return np.sqrt(r2) <= (p.R if radius is None else radius)


def _disc_radius(yloc, zloc, p, dy):
    """Distance from rotor p's yaw-projected axis, in the same metric as _disc_mask."""
    return np.sqrt(((yloc + dy + p.Yoffset) ** 2) / (np.cos(p.beta) ** 2)
                   + (zloc - p.Zhub) ** 2)


def _disc_fraction(yloc, zloc, p, dy, dl_cell, radius=None):
    """Fractional cell coverage of the yaw-projected disc, instead of in/out."""
    rad = p.R if radius is None else radius
    r = _disc_radius(yloc, zloc, p, dy)
    return np.clip(0.5 - (r - rad) / dl_cell, 0.0, 1.0)


def initialize_wake_field(params, stacked, local, companions=(), kw0=None, wage0=None,
                          ti0=None):
    yloc = np.asarray(stacked.yloc)
    zloc = np.asarray(stacked.zloc)
    beta = params.beta

    dl = params.dl
    Uin = np.asarray(local.Uin)
    U = Uin.copy()
    U_pre = Uin.copy()
    _dl_cell = np.sqrt((yloc[1, 0] - yloc[0, 0]) * (zloc[0, 1] - zloc[0, 0]))
    for p, dy in ((params, 0.0),) + tuple(companions):
        frac = _disc_fraction(yloc, zloc, p, dy, _dl_cell)
        a0 = float(0.5 * (1.0 - np.sqrt(max(1.0 - _ct_eff(p) * 0.5, 0.0))))
        U -= 2.0 * U * a0 * frac

    a_n = 0.5 * (1.0 - np.sqrt(max(1.0 - NACELLE_CD, 0.0)))
    cell = (yloc[1, 0] - yloc[0, 0]) * (zloc[0, 1] - zloc[0, 0])
    dl_cell = np.sqrt(cell)
    for p, dy in ((params, 0.0),) + tuple(companions):
        d_n = float(getattr(p.config, 'nacelle_D', 0.0) or 0.0)
        if d_n <= 0.0:
            continue
        r_n = 0.5 * d_n * p.D
        r = np.sqrt(((yloc + dy + p.Yoffset) ** 2) / (np.cos(p.beta) ** 2)
                    + (zloc - p.Zhub) ** 2)
        frac = np.clip(0.5 - (r - r_n) / dl_cell, 0.0, 1.0)
        carved = float(frac.sum()) * cell
        if carved <= 0.0:
            continue
        frac = np.clip(frac * (np.pi * r_n ** 2 * np.cos(p.beta)) / carved, 0.0, 1.0)
        U -= 2.0 * U * a_n * frac

    U_smooth = Uin - smooth_2d(Uin - U, kernel_size=3)

    kw_seed = (np.zeros_like(U_smooth) if kw0 is None
               else np.asarray(kw0).reshape(U_smooth.shape))

    Uw_old = np.maximum(np.asarray(init_Uin(params)) - U_pre, 0.0)
    Uw_new = np.maximum(U_pre - U, 0.0)
    _eps_w = (AGE_BLEND_EPS * params.field_params.I_amb
              * float(nominal_hub_velocity(params)))
    wage_in = (np.zeros_like(U_smooth) if wage0 is None
               else np.asarray(wage0).reshape(U_smooth.shape))
    wage_seed = wage_in * Uw_old / (Uw_old + Uw_new + _eps_w)

    _U0n = float(nominal_hub_velocity(params))
    _kBn = 1.5 * ((params.field_params.I_amb / IU_TO_TI) * _U0n) ** 2
    _kws = np.asarray(kw_seed)
    ti_fresh = np.sqrt(np.maximum(2.0 / 3.0 * (_kBn + _kws), 0.0)) / _U0n
    ti_prev = ti_fresh if ti0 is None else np.asarray(ti0).reshape(ti_fresh.shape)
    ti_seed = ((ti_prev * Uw_old + ti_fresh * (Uw_new + _eps_w))
               / (Uw_old + Uw_new + _eps_w))
    new_stacked = dataclasses.replace(
        stacked,
        U=jnp.asarray(U_smooth),
        kw=jnp.asarray(kw_seed),
        wage=jnp.asarray(wage_seed),
        ti=jnp.asarray(ti_seed),
        X=stacked.X.at[0].set(0.0),
        t=stacked.t.at[0].set(0.0),
    )
    return new_stacked, dl


def calculate_deficit_field(params, local, stacked, dl, max_steps=1500, companions=(),
                            carried=(), n_slots=None, read_at=None):
    """March the deficit field; return its frames, all of them or, given `read_at`, only those that..."""
    yloc = np.asarray(params.yloc)
    zloc = np.asarray(params.zloc)
    rotors = (((params, 0.0, 0.0),)
              + tuple((p, dy, 0.0) for p, dy in companions)
              + tuple(carried))
    n_slots = max(len(rotors) if n_slots is None else n_slots, len(rotors))
    rotor_mask = np.zeros((n_slots,) + yloc.shape)
    Ct_eff = np.zeros(n_slots)
    x_off = np.zeros(n_slots)
    dl_cell = np.sqrt((yloc[1, 0] - yloc[0, 0]) * (zloc[0, 1] - zloc[0, 0]))
    for i, (p, dy, off) in enumerate(rotors):
        if off >= RAMP_REACH_D * p.D:
            continue
        rotor_mask[i] = np.asarray(smooth_2d(_disc_fraction(yloc, zloc, p, dy, dl_cell),
                                             kernel_size=3))
        Ct_eff[i] = _ct_eff(p)
        x_off[i] = off

    ti_seed = np.asarray(stacked.ti)
    if ti_seed.ndim == 3:
        ti_seed = ti_seed[0]
    if ti_seed.size == 0:
        _U0 = float(nominal_hub_velocity(params))
        _kB = 1.5 * ((params.field_params.I_amb / IU_TO_TI) * _U0) ** 2
        ti_seed = np.full_like(yloc, np.sqrt(2.0 / 3.0 * _kB) / _U0)
    stacked = dataclasses.replace(stacked, ti=jnp.asarray(ti_seed))

    adapter = DeficitFieldConfig(
        pos=jnp.asarray(params.pos), D=params.D, Uhub=local.Uhub, Uin=jnp.asarray(local.Uin), Zhub=params.Zhub,
        U0=float(nominal_hub_velocity(params)),
        rotor_mask=jnp.asarray(rotor_mask),
        Ct_eff=jnp.asarray(Ct_eff),
        x_off=jnp.asarray(x_off),
    )
    dt_cap = min(dl / local.Uhub, 0.25 * params.D / local.Uhub)

    seed = _frame(stacked, 0, unstacked=_UNSTACKED_MEMBERS + ("U", "kw", "wage", "ti"))
    grid = (np.asarray(seed.yloc), np.asarray(seed.zloc))
    head = _vortex_field_to_numpy(seed, grid)
    if read_at is not None:
        if len(read_at) > 1:
            raise ValueError("a lean march keeps the frames for one read position at most")
        x_read = float(read_at[0]) if read_at else np.inf
        X, slots, rows, was_active = _calculate_deficit_field_lean_jit(
            stacked, seed, adapter, params.field_params.WV, dl, dt_cap,
            local.calculation_domain, max_steps, _below_threshold(x_read, jnp.result_type(seed.X)))
        return _frames_read_at(head, X, slots, rows, int(jnp.sum(was_active)), grid, x_read)

    stacked_out, was_active = _calculate_deficit_field_jit(
        stacked, seed, adapter, params.field_params.WV,
        dl, dt_cap,
        local.calculation_domain, max_steps,
    )
    kept_count = int(jnp.sum(was_active)) + 1
    return [head] + _stacked_frames_to_numpy(stacked_out, kept_count - 1, grid)


def _deficit_advance(stacked, adapter, WV, dl, dt_cap, calculation_domain):
    """One deficit-march step, `(current, active) -> ((new, still_active), frame)`."""
    def advance(carry):
        current, _ = carry

        age_D = jnp.maximum(current.wage / adapter.D, 1e-6)
        NuT = du_nu_t(current.ti, age_D, adapter.U0, adapter.D)
        nut_bud = NuT
        _dz = current.zloc[0, 1] - current.zloc[0, 0]
        _inv2d = 1.0 / dl ** 2 + 1.0 / _dz ** 2
        _Urep = representative_velocity(current.U, adapter.Uin)
        _ratio = NuT / jnp.maximum(current.U, velocity_floor(adapter.Uin))
        _nu_over_u = jnp.max(_ratio[1:-1, 1:-1])
        dt = jnp.minimum(dt_cap, DIFFUSIVE_CFL
                         / (2.0 * _inv2d * _Urep * _nu_over_u + 1e-12))

        new = interpolate_vec_data(stacked, current.t + dt)
        U, X_new, V_tot, W_tot = advance_wake_field(current, dt, NuT, adapter, WV)
        kw_new = advance_kw(current.kw, current.U, adapter.Uin, current.yloc,
                            current.zloc, X_new - current.X, nut_bud,
                            du_eps_coef(current.ti, adapter.D), V_tot, W_tot)
        ti_new = advect_incident_ti(current.ti, current.U, adapter.Uin, current.yloc,
                                    current.zloc, X_new - current.X, V_tot, W_tot,
                                    nu_t=nut_bud)

        age_old = (current.X + adapter.x_off) / adapter.D
        age_new = (X_new + adapter.x_off) / adapter.D
        a_old = axial_induction_ramped(adapter.Ct_eff, age_old)
        a_new = axial_induction_ramped(adapter.Ct_eff, age_new)
        ramp = (1.0 - 2.0 * a_new) / jnp.maximum(1.0 - 2.0 * a_old, 1e-6)
        U = U * jnp.prod(1.0 + (ramp[:, None, None] - 1.0) * adapter.rotor_mask,
                         axis=0)

        wage_new = current.wage + (X_new - current.X)
        new = dataclasses.replace(new, U=U, kw=kw_new, wage=wage_new, ti=ti_new,
                                  X=X_new, t=current.t + dt)

        still_active = new.X <= calculation_domain
        emitted = dataclasses.replace(new, yloc=_UNSTACKED, zloc=_UNSTACKED)
        return (new, still_active), emitted

    return advance


@partial(jax.jit, static_argnames=("max_steps",))
def _calculate_deficit_field_jit(stacked, seed, adapter, WV,
                                  dl, dt_cap, calculation_domain, max_steps):
    advance = _deficit_advance(stacked, adapter, WV, dl, dt_cap, calculation_domain)
    # The caller reads only the active prefix, so the unused tail is left as zeros.
    stacked_out, was_active, _ = march_until_stopped(advance, (seed, True), max_steps)
    stacked_out = dataclasses.replace(stacked_out, yloc=seed.yloc, zloc=seed.zloc)
    return stacked_out, was_active


@partial(jax.jit, static_argnames=("max_steps",))
def _calculate_deficit_field_lean_jit(stacked, seed, adapter, WV,
                                       dl, dt_cap, calculation_domain, max_steps, x_read):
    """The same march, keeping every step's X but only three frames:"""
    advance = _deficit_advance(stacked, adapter, WV, dl, dt_cap, calculation_domain)
    frame_aval = jax.eval_shape(lambda c: advance(c)[1], (seed, True))

    def advance_lean(carry):
        current, slots, rows, row, _ = carry
        (new, still_active), frame = advance((current, True))
        s = jnp.where(frame.X < x_read, 0, jnp.where(rows[1] < 0, 1, 2))
        slots = jax.tree.map(lambda buf, x: buf.at[s].set(x), slots, frame)
        return (new, slots, rows.at[s].set(row), row + 1, still_active), frame.X

    init = (seed, jax.tree.map(lambda a: jnp.zeros((3,) + a.shape, a.dtype), frame_aval),
            jnp.full(3, -1, jnp.int32), jnp.int32(0), True)
    X, was_active, (_, slots, rows, _, _) = march_until_stopped(advance_lean, init, max_steps)
    return X, slots, rows, was_active


def _vortex_field_to_numpy(vortex_field, grid=None):
    """Converts every array field of a VortexField from jnp back to plain numpy, and t/X (0-d jnp..."""
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
        wage=np.asarray(vortex_field.wage), ti=np.asarray(vortex_field.ti),
        t=float(vortex_field.t), X=float(vortex_field.X),
    )


FRAME_BUCKET = 256


def _stacked_frames_to_numpy(stacked, n, grid):
    """The first `n` frames of a stacked march as numpy VortexFields, in ONE transfer."""
    # Slice on the device to `n` rounded up to a bucket and trim on the host: `n` changes on
    # every evaluation, and an exact [:n] compiles a fresh slice per field every time.
    m = -(-n // FRAME_BUCKET) * FRAME_BUCKET
    host = jax.device_get({f.name: getattr(stacked, f.name)[:m]
                           for f in dataclasses.fields(stacked)
                           if f.name not in _UNSTACKED_MEMBERS})
    host = {k: v[:n] for k, v in host.items()}
    yloc, zloc = grid
    scalars = ("t", "X")
    return [dataclasses.replace(
                stacked, yloc=yloc, zloc=zloc,
                t=float(host["t"][i]), X=float(host["X"][i]),
                **{k: v[i] for k, v in host.items() if k not in scalars})
            for i in range(n)]


def _below_threshold(x, dtype):
    """The value that splits frames at `x` in the march's own precision:"""
    dtype = np.dtype(dtype)
    if not np.isfinite(x) or dtype == np.float64:
        return np.asarray(x, dtype)
    c = np.float32(x)
    return np.nextafter(c, np.float32(np.inf)) if c < x else c


def _frames_read_at(head, X_steps, slots, slot_rows, n, grid, x):
    """`head` and the frames of an `n`-step lean march that reading it at `x` touches, in order:"""
    X = np.concatenate([[float(head.X)], np.asarray(jax.device_get(X_steps))[:n].astype(float)])
    keep = {0, len(X) - 1}
    if np.isfinite(x):
        i1 = int(np.searchsorted(X, x))
        keep |= {min(max(i, 0), len(X) - 1) for i in (i1 - 1, i1)}
        keep.add(int(np.argmin(np.abs(X - x))))
    rows = sorted(i - 1 for i in keep if i > 0)
    host, slot_rows = jax.device_get((slots, slot_rows))
    slot_of = {int(r): k for k, r in enumerate(slot_rows) if r >= 0}
    missing = [r for r in rows if r not in slot_of]
    if missing:
        raise RuntimeError(f"lean deficit march did not keep step(s) {missing} that a read at "
                           f"x={x:.4f} needs; run with all_frames=True")
    yloc, zloc = grid
    return [head] + [dataclasses.replace(
                host, yloc=yloc, zloc=zloc,
                t=float(host.t[slot_of[r]]), X=float(host.X[slot_of[r]]),
                **{f.name: getattr(host, f.name)[slot_of[r]] for f in dataclasses.fields(host)
                   if f.name not in _UNSTACKED_MEMBERS + ("t", "X")})
            for r in rows]
