"""Single marched field through the whole farm -- Zong & Porte-Agel (2020) p.22.

One velocity field and one accumulating vortex cloud are carried from rotor to rotor, so
the model "requires no additional wake superposition laws". Each rotor's deficit is an
initial condition added to the running field, and its shed vortices are appended to the
still-evolving cloud rather than starting a fresh, isolated system.
"""
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

VOR_FULL, DEF_FULL = 1000, 1500     # scan lengths; see _march_segment
TRUNCATE_DOMAIN = True              # False marches every rotor over the full domain
X_STATION_TOL_D = 0.25              # rotors within this many D in x shed together


def _splice(a, b):
    """Interleave two [real | mirror] vortex arrays into one [all real | all mirror] array.

    `_merge_close_vortices` treats the first half of every array as the real vortices and
    regenerates the second half from it each step. A plain concatenation would give
    [C_real, C_mirror, F_real, F_mirror], whose first half is (C_real, C_mirror) -- so the
    fresh rotor's vortices are overwritten by a negated copy of the carried ones on the
    first step of the march. Splice the halves instead.
    """
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
                   carried=(), n_slots=None, kw0=None, wage0=None, ti0=None):
    """March one turbine only as far as its field is actually read.

    Measured on the 8-turbine wind tunnel, one process, identical results throughout:
        full domain                139.0 s
        domain truncated            61.9 s   <- 2.2x, this
        + scan budgets sized to it  81.3 s   <- slower again, so do NOT do that
    Shrinking the scan lengths to match backfires: steps past the natural stop re-emit a
    frozen state and cost almost nothing (measured flat from 100 to 1000 steps once the
    cloud has frozen), while any budget that turns out short costs a whole extra march.
    """
    if TRUNCATE_DOMAIN:
        t.calculation_domain = need_m
    need_m = min(need_m, t.calculation_domain)

    t.Uin = U0                       # initialize_wake_field carves the top-hat out of this
    t.simulate_vortex_field(seed=seed, total_steps=VOR_FULL)
    t.initialize_wake_field(companions, kw0=kw0, wage0=wage0, ti0=ti0)
    t.Uin = Uin_global               # solver reference: the GLOBAL profile
    t.calculate_deficit_field(max_steps=_deficit_steps(t),
                              companions=companions, carried=carried, n_slots=n_slots)
    if float(t.wake_field[-1].X) < need_m * 0.999:
        raise RuntimeError(f"march at x={t.pos[0]:.0f} reached only "
                           f"{float(t.wake_field[-1].X):.1f} m of {need_m:.1f} m")

    # The cloud and the deficit do NOT share a streamwise coordinate. vortex_model
    # advances `age += Uhub*dt` while the deficit advances `dx = dt*U_rep(U, Uin)`, and
    # U_rep is smaller inside a wake -- measured age/X = 1.41-1.49 on the full-wake row.
    # The cloud's own stop is t_limit = 1.5*calculation_domain/D in Uhub*t, so that ratio
    # eats directly into the margin, and interpolate_vec_data CLAMPS to the last frame
    # rather than raising: if the cloud stopped first, the cross-flow would silently
    # freeze for the tail of every segment and the wake would stop being steered.
    # Measured margin on the full-wake row is 1.05 / 1.28 / 1.35 -- only 5% on the first
    # rotor, so a slower wake (deeper array, higher Ct) could exhaust it.
    _vt = np.asarray(t.vortex_field.t)
    if _vt.size and float(t.wake_field[-1].t) > float(_vt.max()) * (1.0 + 1e-9):
        raise RuntimeError(
            f"vortex cloud at x={t.pos[0]:.0f} stopped at t={float(_vt.max()):.5f} but the "
            f"deficit march reached t={float(t.wake_field[-1].t):.5f}, so the cross-flow is "
            f"frozen (clamped to the last cloud frame) over the tail of this segment. The "
            f"cloud ages on Uhub while the deficit advances on the local wake speed; raise "
            f"the vortex t_limit in vortex_model.")


def _deficit_steps(t):
    """Scan length for the deficit march, scaled to the grid instead of fixed.

    DEF_FULL was a bare literal, which put a CEILING ON REFINEMENT: the explicit diffusive
    bound goes as dy^2 while the scan length did not, so the steps needed grow as
    n_grids^2 and the march failed above n_grids ~ 44-56 with "reached only X m of Y m".
    A grid-convergence study was therefore structurally impossible above 36, which is
    uncomfortably close to the 20 and 28 the model is validated at.

    Over-allocating is FREE, measured rather than assumed: on the 8-turbine row,
    DEF_FULL = 1500 / 3000 / 6000 runs in 70.5 / 68.0 / 74.2 s -- the ordering is noise --
    and returns a BIT-IDENTICAL T8 (0.650012172 at all three), because an inactive scan
    step short-circuits through `when_inactive`. Under-allocating costs a whole march. So
    the asymmetry says to be generous.

    The literal stays the n_grids = 20 reference, since that is where it was tuned, and
    the quadratic is the bound's own scaling rather than a fitted margin. _march_segment
    still raises if a march falls short, so this is a budget and not a guarantee.
    """
    n = int(getattr(t._params.field_params, "n_grids", 20))
    return int(DEF_FULL * max(1.0, (n / 20.0) ** 2))


def _assert_turbulence_admissible(field_params):
    """Refuse an ambient TI the closure cannot diffuse at, instead of marching to garbage.

    Du Eq (26) carries the prefactor max(kv1*TI - kv0, 0), which is EXACTLY ZERO at and
    below TI = kv0/kv1 = 0.02, i.e. I_amb <= 0.0256. The deficit then has no turbulent
    diffusion at all, and nothing damps the central-differenced cross-plane advection:
    measured on a 2-rotor farm at 25 deg of yaw, the field reaches U = -17.7 m/s while
    the waked rotor still reports a perfectly plausible Uhub/Uh = 0.747. Silent, and the
    sign of the error is invisible in any aggregate.

    Unyawed it happens to stay bounded, because (V, W) is small -- so this cannot be left
    to chance on the yaw angle. test_wide_farm_marches_without_diverging asserts the
    bound but has never swept I_amb.

    Raising rather than flooring nu_t is deliberate: a floor is a fitted constant, and it
    would paper over a regime Eq (26) genuinely does not model. Du calibrated it over
    TI = 0.041-0.071 and it is already an extrapolation at our 0.13; below 0.02 it does
    not merely extrapolate, it returns zero. Note the approach to the cliff is steep --
    nu_t at I_amb = 0.03 is an order of magnitude below its value at 0.077 -- so a run
    just above the threshold is weakly damped rather than safe.
    """
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
    """Nv and merge_threshold are coupled, and at the defaults the ring self-destructs.

    The shed ring is Nv elements evenly spaced in phi on the rotor edge, foreshortened to
    an ellipse of semi-minor axis R*cos(beta). Its CLOSEST neighbour spacing is therefore
    ~ pi*D*cos(beta)/Nv, and _merge_close_vortices (MERGE_CORE_RATIO is None, so the
    legacy absolute criterion is live) merges any pair closer than merge_threshold*D. The
    ring survives only while

        pi * cos(beta) / Nv  >  merge_threshold

    At the defaults Nv = 49 and merge_threshold = 0.05 that fails at beta = 40 deg:
    measured 8 pairs collapse on step one at 40 and 16 at 45, against none at 35. The
    yaw sweep runs to +-40, so the model's yaw response has a discretisation kink inside
    the swept range whose LOCATION depends on Nv -- a knob nobody would think to hold
    fixed while comparing yaw curves. Nv = 36 is clean to 45 deg.

    Raising rather than auto-adjusting: changing Nv or the threshold silently would move
    every result, and which one to trade is a modelling decision. See the note on
    MERGE_CORE_RATIO in vortex_model for the relative criterion that would retire this.
    """
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


def _assert_shared_grid(ts):
    """Every rotor must sit on one absolute cross-plane grid, in BOTH y and z.

    Turbine i's inflow is read by interpolating turbine i-1's field onto i's grid. If
    the two do not coincide, points outside the source grid fall back to the undisturbed
    profile, and a wake can vanish with no error raised -- a 2x2 farm with its columns 7D
    apart once reported every rotor at free-stream. WindFarm builds the shared grid; this
    catches a farm assembled some other way, loudly rather than silently.

    The z half exists because TERRAIN IS NOT SUPPORTED and used to fail silently. The two
    halves of the model disagree about what z means when pos[2] varies: init_Uin builds
    the inflow log profile on the rotor's LOCAL zloc, i.e. terrain-following, while
    field_ops interpolates in ABSOLUTE z (zloc + pos[2]). Config.elevation_func is a
    public input that sets pos[:, 2], so a user could reach this, and the symptom was the
    silent free-stream fallback above rather than an error. Supporting terrain means
    deciding which z the profile belongs in -- a modelling question, not a bug fix -- so
    this refuses the case instead of guessing.
    """
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
    """Group the (x-sorted) turbines into streamwise stations.

    Rotors closer together in x than `tol` are treated as one station and shed
    simultaneously. They must be: the march cuts each segment at the next rotor, and the
    per-segment physics describes ONE rotor -- the near-wake ramp is applied over that
    rotor's disc and Du's Eq (26) measures x from that rotor. A wake handed to a new segment
    therefore stops ramping and has its eddy viscosity restarted, so splitting a march
    changes the answer. Measured on a column with a bystander 7D to the side, which
    should have no effect at all: +17.9% when it sat at the same x (a 0.5D segment),
    -9.3% when it sat at 3D. Grouping restores full-spacing segments, and for a gridded
    farm it is also the physically right statement.
    """
    stations = []
    for t in ts:
        if stations and abs(t.pos[0] - stations[-1][0].pos[0]) <= tol:
            stations[-1].append(t)
        else:
            stations.append([t])
    return stations


def solve_single_field(wf, verbose=False):
    """March one field and one vortex cloud through the farm, station by station."""
    ts = wf.turbines
    _assert_turbulence_admissible(wf.field_params)
    _assert_ring_survives_merging(ts)
    _assert_shared_grid(ts)
    stations = _x_stations(ts, X_STATION_TOL_D * ts[0].D) if ts else []
    prev, carried_vortex, carried_kw = None, None, None
    carried_wage, carried_ti = None, None
    audit = []
    # Rotors already shed, in march order. The ramp stack is padded to the whole farm
    # rather than to this list's current length, so the scan traces once per farm size
    # instead of once per station -- and once per BO iteration rather than per layout.
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
        # k_w is marched like U, so it must be handed over like U. It is read at the SAME
        # frame the next station starts from, so production already deposited upstream is
        # carried rather than restarted -- which is what removes the need for any
        # superposition rule.
        kw0 = np.zeros_like(U0) if carried_kw is None else np.asarray(carried_kw)
        # The shear-layer age crosses with k_w. It MUST, or Du Eq (26) restarts for
        # the whole plane at every station and a rotor that carves nothing in this
        # column still changes it -- measured at a Ct-independent 2.6%.
        wage0 = None if carried_wage is None else np.asarray(carried_wage)
        ti0 = None if carried_ti is None else np.asarray(carried_ti)
        # A rotor is misaligned relative to the flow it sits in, not to the tunnel axis,
        # and a steered upstream wake has already turned that flow. delta is read from the
        # field this station was HANDED, before any of its rotors shed, so there is no
        # circularity. Only the AERODYNAMICS follow beta_aero; Yoffset, the projected disc
        # and the vortex ring keep the commanded angle. See patch_skew2.py.
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

        # every rotor at this station reads its own disc out of the same incoming field
        for t in group:
            dy = lead.pos[1] - t.pos[1]
            # The rotor sits l_n*sin(beta) off the tower axis when yawed, so the disc
            # the power is read over must be shifted by Yoffset exactly as the initial
            # condition, the vortex ring and nominal_hub_velocity are. Without it the
            # two sides of the efficiency ratio were averaged over discs 0.085D apart
            # at beta = 25 deg. Yoffset is zero at zero yaw, so aligned cases are
            # unaffected.
            # ...and by COVERAGE, not in/out by cell centre: see tp.disc_weights_on.
            rw = tp.disc_weights_on(lead.yloc, lead.zloc, t._params, dy)
            t.Uhub = tp.rotor_average(U0, rw)
            # Power is read from the CUBIC rotor-equivalent speed over the same disc
            # (Zong Eq 4.11 / IEC REWS); the aerodynamics keep the arithmetic mean.
            t.Uhub3 = tp.rotor_equivalent_cubic(U0, rw)
            t.V, t.W = zeros, zeros  # upstream influence is explicit in the vortex cloud

        # ... and sheds its ring into the shared cloud, shifted into the lead's frame
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
        # only the LAST station's field is read to the end of the domain
        need_m = (stations[g + 1][0].pos[0] - lead.pos[0] + 0.5 * lead.D
                  if g + 1 < len(stations) else lead.calculation_domain)
        _march_segment(lead, seed, need_m, U0, Uin_global, companions,
                       carried, n_slots, kw0, wage0, ti0)
        shed_before.extend(group)

        # companions share the station's marched field; nothing further is solved on them
        for t in group:
            if t is not lead:
                t.vortex_field, t.dl, t.wake_field = lead.vortex_field, lead.dl, lead.wake_field

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
