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
import numpy as np
import jax.numpy as jnp

from .data_structures import VortexField
from .field_ops import interpolate_local_velocity_field
from . import turbine_physics as tp

VOR_FULL, DEF_FULL = 1000, 1500     # scan lengths; see _march_segment
TRUNCATE_DOMAIN = True              # False marches every rotor over the full domain


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


def _march_segment(t, seed, need_m, up, N_up_max, U0, Uin_global):
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
    t.initialize_wake_field()
    t.Uin = Uin_global               # solver reference: the GLOBAL profile
    t.calculate_deficit_field(up, N_upstream_max=N_up_max, max_steps=DEF_FULL)
    if float(t.wake_field[-1].X) < need_m * 0.999:
        raise RuntimeError(f"march at x={t.pos[0]:.0f} reached only "
                           f"{float(t.wake_field[-1].X):.1f} m of {need_m:.1f} m")


def solve_single_field(wf, verbose=False):
    """March one field and one vortex cloud through the farm, in streamwise order."""
    ts = wf.turbines
    N_up_max = max(len(ts) - 1, 0)
    prev, carried_vortex = None, None
    audit = []

    for i, t in enumerate(ts):
        Uin_global = np.asarray(t.init_Uin())
        R = t.D / 2.0
        rmask = np.sqrt((t.yloc / np.cos(t.beta)) ** 2 + (t.zloc - t.Zhub) ** 2) <= R

        if prev is None:
            U0 = Uin_global.copy()
        else:
            Ud, _ = interpolate_local_velocity_field(
                prev, t.pos[0] - prev.pos[0], t.yloc + t.pos[1], t.zloc + t.pos[2], Uin_global)
            U0 = np.asarray(Ud)

        t.Uhub = float(np.mean(U0[rmask]))
        zeros = np.zeros_like(U0)
        t.V, t.W = zeros, zeros      # upstream influence is explicit in the vortex cloud

        fresh = tp.initial_vortex_state(t._params, t._current_local())
        seed = fresh if carried_vortex is None else _concat_vortices(
            carried_vortex, fresh, zeros, zeros)

        up = [u for u in ts if u.pos[0] < t.pos[0]
              and abs(u.pos[1] - t.pos[1]) < 3 * u.D and abs(u.pos[2] - t.pos[2]) < 3 * u.D]
        # only the LAST rotor's field is read to the end of the domain
        need_m = (ts[i + 1].pos[0] - t.pos[0] + 0.5 * t.D if i + 1 < len(ts)
                  else t.calculation_domain)
        _march_segment(t, seed, need_m, up, N_up_max, U0, Uin_global)

        if verbose:
            audit.append(dict(i=i + 1, x_D=t.pos[0] / t.D, n_vort=int(np.asarray(seed.Y).size),
                              disc=t.Uhub, net_circ=float(np.sum(np.asarray(seed.Circ))),
                              need_D=need_m / t.D))

        if i + 1 < len(ts):
            dx = ts[i + 1].pos[0] - t.pos[0]
            k = int(np.argmin([abs(float(f.X) - dx) for f in t.wake_field]))
            f = t.wake_field[k]
            carried_vortex = VortexField(Y=f.Y, Z=f.Z, Rv=f.Rv, Circ=f.Circ, age=f.age, active=f.active, yloc=jnp.array([]), zloc=jnp.array([]),
                                         V=zeros, W=zeros, OmegaX=jnp.array([]), t=0.0)
        prev = t

    if verbose:
        print("   T   x/D   vortices   net circ    disc U     need")
        for a in audit:
            print(f"  {a['i']:>2}  {a['x_D']:5.1f}   {a['n_vort']:>8}  {a['net_circ']:+.2e}"
                  f"   {a['disc']:6.3f}   {a['need_D']:5.1f}D")
    return wf, audit
