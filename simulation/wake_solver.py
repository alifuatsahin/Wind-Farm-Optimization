"""Single marched field through the whole farm -- Zong & Porte-Agel (2020) p.22."""
import dataclasses

import os
import numpy as np
import jax.numpy as jnp

from .data_structures import VortexField
from .field_ops import (interpolate_local_velocity_field,
                        interpolate_local_lateral_field)
from .turbine_state import CP_YAW_EXP
from .utils import DU_KV0, DU_KV1, IU_TO_TI
from . import turbine_physics as tp

VOR_FULL, DEF_FULL = 1000, 1500
TRUNCATE_DOMAIN = True
X_STATION_TOL_D = 0.25


def _splice(a, b):
    """Interleave two [real | mirror] vortex arrays into one [all real | all mirror] array."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    na, nb = a.shape[0] // 2, b.shape[0] // 2
    return jnp.concatenate([a[:na], b[:nb], a[na:], b[nb:]])


def _concat_vortices(carried, fresh, V_bg, W_bg):
    """One cloud: turbine i+1's fresh ring appended to the still-evolving upstream cloud."""
    return VortexField(
        Y=_splice(carried.Y, fresh.Y),
        Z=_splice(carried.Z, fresh.Z),
        Rv=_splice(carried.Rv, fresh.Rv),
        Circ=_splice(carried.Circ, fresh.Circ),
        age=_splice(carried.age, fresh.age),
        active=_splice(carried.active, fresh.active),
        yloc=jnp.array([]), zloc=jnp.array([]),
        V=jnp.asarray(V_bg), W=jnp.asarray(W_bg), OmegaX=jnp.array([]), t=0.0,
    )


def _march_segment(t, seed, need_m, U0, Uin_global, companions=(),
                   carried=(), n_slots=None, kw0=None, wage0=None, ti0=None, read_at=None):
    """March one turbine only as far as its field is actually read."""
    if TRUNCATE_DOMAIN:
        t.calculation_domain = need_m
    need_m = min(need_m, t.calculation_domain)

    t.Uin = U0
    t.simulate_vortex_field(seed=seed, total_steps=VOR_FULL)
    t.initialize_wake_field(companions, kw0=kw0, wage0=wage0, ti0=ti0)
    t.Uin = Uin_global
    t.calculate_deficit_field(max_steps=_deficit_steps(t),
                              companions=companions, carried=carried, n_slots=n_slots,
                              read_at=read_at)
    if float(t.wake_field[-1].X) < need_m * 0.999:
        raise RuntimeError(f"march at x={t.pos[0]:.0f} reached only "
                           f"{float(t.wake_field[-1].X):.1f} m of {need_m:.1f} m")

    _vt = np.asarray(t.vortex_field.t)
    if _vt.size and float(t.wake_field[-1].t) > float(_vt.max()) * (1.0 + 1e-9):
        _cause = (f"it EXHAUSTED its step budget of VOR_FULL={VOR_FULL} without ever "
                  f"reaching its natural stop, so raise VOR_FULL"
                  if t.vortex_exhausted else
                  f"it reached its natural t_limit stop with {int(np.sum(np.diff(_vt) > 0)) + 1} "
                  f"of VOR_FULL={VOR_FULL} steps used, so raise the vortex t_limit in "
                  f"vortex_model")
        raise RuntimeError(
            f"vortex cloud at x={t.pos[0]:.0f} stopped at t={float(_vt.max()):.5f} but the "
            f"deficit march reached t={float(t.wake_field[-1].t):.5f}, so the cross-flow is "
            f"frozen (clamped to the last cloud frame) over the tail of this segment. The "
            f"cloud ages on Uhub while the deficit advances on the local wake speed, and "
            f"{_cause}.")


def _deficit_steps(t):
    """Scan length for the deficit march, scaled to the grid instead of fixed."""
    n = int(getattr(t._params.field_params, "n_grids", 20))
    return int(DEF_FULL * max(1.0, (n / 20.0) ** 2))


def _assert_turbulence_admissible(field_params):
    """Refuse an ambient TI the closure cannot diffuse at, instead of marching to garbage."""
    ti = float(field_params.I_amb) / IU_TO_TI
    ti_min = DU_KV0 / DU_KV1
    if ti <= ti_min:
        raise ValueError(
            f"I_amb = {field_params.I_amb:.4f} gives ambient TI = {ti:.4f}, at or below "
            f"the TI = {ti_min:.3f} (I_amb = {ti_min * IU_TO_TI:.4f}) where Du Eq (26) "
            f"returns exactly zero eddy viscosity. The deficit march has no diffusion to "
            f"damp its cross-plane advection there and diverges silently. Raise I_amb "
            f"above {ti_min * IU_TO_TI:.4f}, or use a closure valid at low ambient "
            f"turbulence -- this one is calibrated over TI = 0.041-0.071.")


def _assert_ring_survives_merging(ts):
    """Nv and merge_threshold are coupled, and at the defaults the ring self-destructs."""
    for t in ts:
        beta = abs(float(t._params.beta))
        nv = int(t._params.Nv)
        thr = float(t._params.field_params.merge_threshold)
        spacing = np.pi * np.cos(beta) / nv
        if spacing <= thr:
            raise ValueError(
                f"rotor at x={t.pos[0]:.1f} yawed {np.degrees(beta):.0f} deg has ring "
                f"spacing {spacing:.4f} D at Nv={nv}, at or below merge_threshold="
                f"{thr:.4f} D, so adjacent ring elements merge on the first step and the "
                f"ring collapses. Lower Nv (36 is clean to 45 deg), or lower "
                f"merge_threshold below {spacing:.4f}.")


def _assert_domain_reaches_ground(ts):
    """Refuse a vertical domain whose bottom is not the ground, instead of pinning the deficit to zero..."""
    for t in ts:
        need = 2.0 * t.Zhub / t.D
        if t._params.field_params.max_Z < need:
            raise ValueError(
                f"max_Z = {t._params.field_params.max_Z:.3f} puts the bottom of the "
                f"vertical domain at z = {t.Zhub - t._params.field_params.max_Z * t.D / 2:.2f} m, "
                f"above the ground, for a rotor at Zhub = {t.Zhub:.1f} m and D = {t.D:.1f} m. "
                f"The deficit is pinned to zero on that row and the k_w budget treats it as "
                f"the ground, so the wake would be sliced off in mid-air. Use "
                f"max_Z > 2*Zhub/D = {need:.3f}.")


def _assert_shared_grid(ts):
    """Every rotor must sit on one absolute cross-plane grid, in BOTH y and z."""
    if len(ts) < 2:
        return
    for axis, loc, off, name in (("lateral", "yloc", 1, "y_bounds"),
                                 ("vertical", "zloc", 2, "elevation")):
        ref = np.asarray(getattr(ts[0], loc)) + ts[0].pos[off]
        for t in ts[1:]:
            cur = np.asarray(getattr(t, loc)) + t.pos[off]
            if cur.shape != ref.shape or not np.allclose(cur, ref, rtol=0, atol=1e-9):
                raise ValueError(
                    f"turbines do not share one absolute {axis} grid, so the marched "
                    f"field cannot carry a wake between them. "
                    + ("Build the farm through WindFarm, which spans the layout's lateral "
                       "extent; constructing Turbine objects directly without y_bounds "
                       "gives each its own grid." if off == 1 else
                       "This is almost always a non-constant pos[:, 2] from "
                       "Config.elevation_func: TERRAIN IS NOT SUPPORTED, because the "
                       "inflow profile is built on terrain-following z while the field is "
                       "interpolated in absolute z. Set a constant elevation."))


def _x_stations(ts, tol):
    """Group the (x-sorted) turbines into streamwise stations."""
    stations = []
    for t in ts:
        if stations and abs(t.pos[0] - stations[-1][0].pos[0]) <= tol:
            stations[-1].append(t)
        else:
            stations.append([t])
    return stations


def solve_single_field(wf, verbose=False, all_frames=True):
    """March one field and one vortex cloud through the farm, station by station."""
    ts = wf.turbines
    _assert_turbulence_admissible(wf.field_params)
    _assert_ring_survives_merging(ts)
    _assert_domain_reaches_ground(ts)
    _assert_shared_grid(ts)
    stations = _x_stations(ts, X_STATION_TOL_D * ts[0].D) if ts else []
    prev, carried_vortex, carried_kw = None, None, None
    carried_wage, carried_ti = None, None
    audit = []
    shed_before = []
    n_slots = max(len(ts), 1)

    for g, group in enumerate(stations):
        lead = group[0]
        Uin_global = np.asarray(lead.init_Uin())

        if prev is None:
            U0 = Uin_global.copy()
        else:
            Ud, _ = interpolate_local_velocity_field(
                prev, lead.pos[0] - prev.pos[0],
                lead.yloc + lead.pos[1], lead.zloc + lead.pos[2], Uin_global)
            U0 = np.asarray(Ud)
        V0 = (np.zeros_like(U0) if prev is None else
              np.asarray(interpolate_local_lateral_field(
                  prev, lead.pos[0] - prev.pos[0],
                  lead.yloc + lead.pos[1], lead.zloc + lead.pos[2])))
        zeros = np.zeros_like(U0)
        kw0 = np.zeros_like(U0) if carried_kw is None else np.asarray(carried_kw)
        wage0 = None if carried_wage is None else np.asarray(carried_wage)
        ti0 = None if carried_ti is None else np.asarray(carried_ti)
        if prev is not None:
            for t in group:
                dy_s = lead.pos[1] - t.pos[1]
                w_s = tp.disc_weights_on(lead.yloc, lead.zloc, t._params, dy_s)
                if w_s.sum() <= 0.0:
                    continue
                Ubar = tp.rotor_average(U0, w_s)
                Vbar = tp.rotor_average(V0, w_s)
                sk = float(np.arctan2(Vbar, max(Ubar, 1e-6)))
                t._params = dataclasses.replace(
                    t._params, skew=sk,
                    Cp=float(t._params.config.Cp) * np.cos(t._params.beta - sk) ** CP_YAW_EXP)

        for t in group:
            dy = lead.pos[1] - t.pos[1]
            rw = tp.disc_weights_on(lead.yloc, lead.zloc, t._params, dy)
            t.Uhub = tp.rotor_average(U0, rw)
            t.Uhub3 = tp.rotor_equivalent_cubic(U0, rw)
            t.V, t.W = zeros, zeros

        fresh = None
        for t in group:
            ring = tp.initial_vortex_state(t._params, t._current_local())
            if t is not lead:
                ring = dataclasses.replace(ring, Y=ring.Y + (t.pos[1] - lead.pos[1]))
            fresh = ring if fresh is None else _concat_vortices(fresh, ring, zeros, zeros)
        seed = fresh if carried_vortex is None else _concat_vortices(
            carried_vortex, fresh, zeros, zeros)

        companions = tuple((t._params, lead.pos[1] - t.pos[1]) for t in group if t is not lead)
        carried = tuple((t._params, lead.pos[1] - t.pos[1], lead.pos[0] - t.pos[0])
                        for t in shed_before)
        need_m = (stations[g + 1][0].pos[0] - lead.pos[0] + 0.5 * lead.D
                  if g + 1 < len(stations) else lead.calculation_domain)
        # The next station reads this march at one x: its distance from here.
        read_at = None if all_frames else (
            (stations[g + 1][0].pos[0] - lead.pos[0],) if g + 1 < len(stations) else ())
        _march_segment(lead, seed, need_m, U0, Uin_global, companions,
                       carried, n_slots, kw0, wage0, ti0, read_at)
        shed_before.extend(group)

        for t in group:
            if t is not lead:
                t.vortex_field, t.dl, t.wake_field = lead.vortex_field, lead.dl, lead.wake_field
            if not all_frames:
                # Only this station's deficit march reads its vortex stack, and holding every
                # station's on the device is ~1 GB each on an 80-rotor farm's grid.
                t.vortex_field = None

        if verbose:
            audit.append(dict(i=g + 1, x_D=lead.pos[0] / lead.D, n=len(group),
                              n_vort=int(np.asarray(seed.Y).size),
                              disc=lead.Uhub, net_circ=float(np.sum(np.asarray(seed.Circ))),
                              need_D=need_m / lead.D))

        if g + 1 < len(stations):
            dx = stations[g + 1][0].pos[0] - lead.pos[0]
            k = int(np.argmin([abs(float(f.X) - dx) for f in lead.wake_field]))
            f = lead.wake_field[k]
            carried_vortex = VortexField(Y=f.Y, Z=f.Z, Rv=f.Rv, Circ=f.Circ, age=f.age,
                                         active=f.active, yloc=jnp.array([]), zloc=jnp.array([]),
                                         V=zeros, W=zeros, OmegaX=jnp.array([]), t=0.0)
            carried_kw = np.asarray(f.kw) if np.size(f.kw) else None
            carried_wage = np.asarray(f.wage) if np.size(f.wage) else None
            carried_ti = np.asarray(f.ti) if np.size(f.ti) else None
        prev = lead

    if verbose:
        print("   S   x/D  rotors  vortices   net circ    disc U     need")
        for a in audit:
            print(f"  {a['i']:>2}  {a['x_D']:5.1f}  {a['n']:>5}  {a['n_vort']:>8}"
                  f"  {a['net_circ']:+.2e}   {a['disc']:6.3f}   {a['need_D']:5.1f}D")
    return wf, audit
