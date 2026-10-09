"""Single marched field through the whole farm -- Zong & Porte-Agel (2020) p.22.

One velocity field and one accumulating vortex cloud are carried from rotor to rotor, so
the model "requires no additional wake superposition laws". Each rotor's deficit is an
initial condition added to the running field, and its shed vortices are appended to the
still-evolving cloud rather than starting a fresh, isolated system.

Two consequences shape the implementation:

  * The solver's reference `Uin` is the GLOBAL, y-independent log profile. That is the
    configuration Eq (3.7) actually assumes, and it makes d2Uin/dy2 identically zero, so
    the correction that a superposed inflow needs cannot arise here at all.
  * Turbine i's field is only ever read as far as turbine i+1, so its domain is cut to one
    spacing. Superposition cannot do this -- there every wake is superposed over the whole
    downstream domain -- and it is where most of the cost difference between the two
    architectures comes from.
"""
import dataclasses

import numpy as np
import jax.numpy as jnp

from .data_structures import VortexField
from .field_ops import interpolate_local_velocity_field
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


def _march_segment(t, seed, need_m, up, N_up_max, U0, Uin_global, companions=()):
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
    t.initialize_wake_field(companions)
    t.Uin = Uin_global               # solver reference: the GLOBAL profile
    t.calculate_deficit_field(up, N_upstream_max=N_up_max, max_steps=DEF_FULL,
                              companions=companions)
    if float(t.wake_field[-1].X) < need_m * 0.999:
        raise RuntimeError(f"march at x={t.pos[0]:.0f} reached only "
                           f"{float(t.wake_field[-1].X):.1f} m of {need_m:.1f} m")


def _assert_shared_grid(ts):
    """Every rotor must sit on one absolute cross-plane grid.

    Turbine i's inflow is read by interpolating turbine i-1's field onto i's grid. If
    the two do not coincide, points outside the source grid fall back to the undisturbed
    profile, and a wake can vanish with no error raised -- a 2x2 farm with its columns 7D
    apart once reported every rotor at free-stream. WindFarm builds the shared grid; this
    catches a farm assembled some other way, loudly rather than silently.
    """
    if len(ts) < 2:
        return
    ref = np.asarray(ts[0].yloc) + ts[0].pos[1]
    for t in ts[1:]:
        cur = np.asarray(t.yloc) + t.pos[1]
        if cur.shape != ref.shape or not np.allclose(cur, ref, rtol=0, atol=1e-9):
            raise ValueError(
                "turbines do not share one absolute lateral grid, so the marched field "
                "cannot carry a wake between them. Build the farm through WindFarm, "
                "which spans the layout's lateral extent; constructing Turbine objects "
                "directly without y_bounds gives each its own grid.")


def _x_stations(ts, tol):
    """Group the (x-sorted) turbines into streamwise stations.

    Rotors closer together in x than `tol` are treated as one station and shed
    simultaneously. They must be: the march cuts each segment at the next rotor, and the
    per-segment physics describes ONE rotor -- the near-wake ramp is applied over that
    rotor's disc and NuT_model measures x from that rotor. A wake handed to a new segment
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
    _assert_shared_grid(ts)
    N_up_max = max(len(ts) - 1, 0)
    stations = _x_stations(ts, X_STATION_TOL_D * ts[0].D) if ts else []
    prev, carried_vortex = None, None
    audit = []

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
        zeros = np.zeros_like(U0)

        # every rotor at this station reads its own disc out of the same incoming field
        for t in group:
            dy = lead.pos[1] - t.pos[1]
            # The rotor sits l_n*sin(beta) off the tower axis when yawed, so the disc
            # the power is read over must be shifted by Yoffset exactly as the initial
            # condition, the vortex ring and nominal_hub_velocity are. Without it the
            # two sides of the efficiency ratio were averaged over discs 0.085D apart
            # at beta = 25 deg. Yoffset is zero at zero yaw, so aligned cases are
            # unaffected.
            rmask = np.sqrt(((lead.yloc + dy + t.Yoffset) / np.cos(t.beta)) ** 2
                            + (lead.zloc - t.Zhub) ** 2) <= t.D / 2.0
            t.Uhub = float(np.mean(U0[rmask]))
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
        up = [u for u in ts if u.pos[0] < lead.pos[0] - X_STATION_TOL_D * u.D
              and abs(u.pos[1] - lead.pos[1]) < 3 * u.D and abs(u.pos[2] - lead.pos[2]) < 3 * u.D]
        # only the LAST station's field is read to the end of the domain
        need_m = (stations[g + 1][0].pos[0] - lead.pos[0] + 0.5 * lead.D
                  if g + 1 < len(stations) else lead.calculation_domain)
        _march_segment(lead, seed, need_m, up, N_up_max, U0, Uin_global, companions)

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
        prev = lead

    if verbose:
        print("   S   x/D  rotors  vortices   net circ    disc U     need")
        for a in audit:
            print(f"  {a['i']:>2}  {a['x_D']:5.1f}  {a['n']:>5}  {a['n_vort']:>8}"
                  f"  {a['net_circ']:+.2e}   {a['disc']:6.3f}   {a['need_D']:5.1f}D")
    return wf, audit
