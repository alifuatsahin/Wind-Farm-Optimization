import dataclasses
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .data_structures import VortexField

# Placeholder emitted in place of the stacked history's constant members, so lax.scan
# stacks a (total_steps, 0) array instead of (total_steps, ny, nz). See the note at the
# end of _simulate_vortex_evolution_jit.
_UNSTACKED = jnp.zeros((0,))


@partial(jax.jit, static_argnames=("total_steps",))
def _simulate_vortex_evolution_jit(config, merge_threshold, cfl_factor, total_steps,
                                   seed=None):
    D = config.D
    Uhub = config.Uhub
    # Dimensionless domain-exit threshold, in Uhub*t/D. It must cover the DEFICIT march,
    # and the two did not share a coordinate: the cloud USED TO age on Uhub while the
    # deficit advances on the local wake speed, which put age/X at 1.41-1.49 on the
    # full-wake row. That is fixed -- the cores now age on config.U_conv -- but this
    # threshold is unchanged and still stated in Uhub*t, because it is a TIME budget for
    # covering the deficit march, not a statement about ageing. The two are independent:
    # the stop is on t, and so is the coverage assertion in _march_segment.
    #
    # At the old 1.5 that margin was spent exactly, and the ALIGNED case -- whose wake is
    # deeper, hence slower, hence ages fastest per unit X -- came out at 0.91-0.92, i.e.
    # the cloud stopped BEFORE the deficit did and interpolate_vec_data silently clamped
    # the cross-flow to the last cloud frame over the final ~8% of rotor 1's segment. The
    # yawed cases sat at 1.04-1.05 and were fine, so the artefact hit the baseline of
    # every steering comparison and not the yawed numerator.
    #
    # 2.5 cleared the worst measured ratio with room for a slower wake still (deeper
    # array, higher Ct). Overshoot is cheap: the march stops on this threshold, and steps
    # past a frozen cloud re-emit a frozen state. wake_solver._march_segment asserts the
    # coverage rather than trusting this constant, which is what makes it safe to leave
    # the constant alone through a change to the ageing.
    t_limit = 2.5 * config.calculation_domain / D

    if seed is None:
        seed = _define_location(config)[0]
    seed = dataclasses.replace(seed, yloc=config.yloc, zloc=config.zloc, OmegaX=jnp.zeros_like(config.yloc))

    def scan_step(carry, _):
        current_state, t, active = carry

        def when_active(_):
            current2 = _vortex2velocity(current_state, config)
            dY, dZ, dist2 = _get_relative_geometry(current2.Y, current2.Z, current2.active)
            Circ, Rv = current2.Circ, current2.Rv
            V_induced, W_induced = _compute_mutual_induction(Circ, Rv, dY, dZ, dist2)
            dt = _calculate_time_step(V_induced, W_induced, cfl_factor, dist2)
            t_next = t + dt
            would_stop = t_next * Uhub / D > t_limit

            def do_advance(_):
                new_Y, new_Z = RK4_step(current2.Y, current2.Z, V_induced, W_induced, Circ, Rv, dt, current2.active)
                # age advances with the convected distance; Rv^2 += 4*nu*dt then integrates
                # to Rv^2 = Ri^2 + 4*eta^2(x), matching Shapiro Eq (4.3).
                # On the WAKE's convection speed, not the hub speed: age is a length
                # and the vortex travels with the wake. config.U_conv = (1-a)*Uhub, which
                # is the closed form of Zong's U_m under the model's own 20% deficit
                # threshold -- see vortex_adapter. Measured age/X on the full-wake row was
                # 1.401/1.101/1.029 before, worst at the FIRST rotor where Uhub is the
                # free stream while the wake convects at ~0.6 of it.
                new_age = current2.age + config.U_conv * dt
                nu = config.nu_const + config.nu_slope * 0.5 * (current2.age + new_age)
                new_Rv = jnp.sqrt(Rv**2 + 4 * nu * dt)
                new_vf = VortexField(
                    Y=new_Y, Z=new_Z, Rv=new_Rv, age=new_age,
                    Circ=current2.Circ, active=current2.active, t=t_next,
                    yloc=current2.yloc, zloc=current2.zloc,
                    V=jnp.zeros_like(current2.V), W=jnp.zeros_like(current2.W), OmegaX=current2.OmegaX
                )
                merged = _merge_close_vortices(new_vf, merge_threshold, dist2)
                merged = dataclasses.replace(merged, t=t_next)
                return merged, t_next, True  # next carry state, active stays True

            def do_freeze(_):
                return current2, t_next, False  # natural stop; current2 is final

            next_state, next_t, next_active = lax.cond(would_stop, do_freeze, do_advance, operand=None)
            return (next_state, next_t, next_active), current2

        def when_inactive(_):
            # No-op past the natural stop: re-emit the already-frozen (already-backfilled)
            # state, no recomputation -- current_state IS current2 from the freezing step.
            return (current_state, t, active), current_state

        next_carry, emitted = lax.cond(active, when_active, when_inactive, operand=None)
        emitted = dataclasses.replace(emitted, yloc=_UNSTACKED, zloc=_UNSTACKED,
                                      U=_UNSTACKED)
        return next_carry, (emitted, active)

    init_carry = (seed, 0.0, True)
    _, (stacked, was_active) = lax.scan(scan_step, init_carry, xs=None, length=total_steps)
    # U is never written by this march; initialize_wake_field supplies the real one.
    stacked = dataclasses.replace(stacked, yloc=config.yloc, zloc=config.zloc,
                                  U=_UNSTACKED)
    return stacked, was_active

def _vortex2velocity(data, config):
    """Returns a NEW VortexField (dataclasses.replace) instead of mutating `data` in place --
    mutate-and-return on a pytree instance isn't meaningful under jit/lax.scan tracing, so this
    is written functionally now even though eager execution wouldn't have required it."""
    Y = data.Y
    Z = data.Z
    Circ = data.Circ
    Rv = data.Rv
    yloc = config.yloc
    zloc = config.zloc

    if getattr(config, "pm_Av", None) is None:
        V, W = _oseenlamb(Circ, Y, Z, Rv, yloc, zloc)
    else:
        V, W = _oseenlamb_pm(Circ, Y, Z, Rv, yloc, zloc,
                             config.pm_Av, config.pm_Aw, config.pm_lo, config.pm_lnr)
    V = V + data.V
    W = W + data.W

    _, dVdZ = jnp.gradient(V, yloc[:, 0], zloc[0, :])
    dWdY, _ = jnp.gradient(W, yloc[:, 0], zloc[0, :])
    OmegaX = dWdY - dVdZ

    return dataclasses.replace(data, OmegaX=OmegaX, yloc=yloc, zloc=zloc, V=V, W=W)

def _define_location(config):
    # 1. Initial ring vortices
    Y = config.D / 2 * jnp.cos(config.phi) * jnp.cos(config.beta) - config.Yoffset
    Z = config.D / 2 * jnp.sin(config.phi) + config.Zhub
    Rv = jnp.full(config.Nv, 0.05 * config.D)
    Circ = jnp.asarray(config.dgamma)

    # 2. Add hub vortex (center)
    Y = jnp.append(Y, -config.Yoffset)
    Z = jnp.append(Z, config.Zhub)
    Rv = jnp.append(Rv, 0.15 * config.D)
    Circ = jnp.append(Circ, -jnp.sum(config.dgamma))

    # 3. Mirror vortices against the wall (ground)
    Y = jnp.concatenate([Y, Y])
    Z = jnp.concatenate([Z, -Z])
    Rv = jnp.concatenate([Rv, Rv])
    Circ = jnp.concatenate([Circ, -Circ])
    active = jnp.ones_like(Y, dtype=bool)  # padding capacity for the whole run: Nv+1 real vortices

    # 4. Vortex age (nu_T grows with it)
    age = jnp.zeros_like(Y)        # distance travelled since shed; sets nu_T(x)

    # 5. Create VortexField object
    vordata = VortexField(
        Y=Y,
        Z=Z,
        Rv=Rv,
        Circ=Circ,
        age=age,
        active=active,
        yloc=jnp.array([]),
        zloc=jnp.array([]),
        V=config.V,
        W=config.W,
        OmegaX=jnp.array([]),
        t=0.0
    )
    return [vordata] # return a list of vordata for extensibility

def _compute_mutual_induction(Circ, Rv, dY, dZ, dist2):
    """Compute induced velocities using Biot-Savart law with core correction."""
    velocity_magnitude = (Circ[jnp.newaxis, :] / (2 * jnp.pi * dist2) *
                        (1 - jnp.exp(-dist2 / (Rv[jnp.newaxis, :]**2))))
    velocity_magnitude = jnp.nan_to_num(velocity_magnitude)

    # Compute velocity components
    V = jnp.nan_to_num(-dZ * velocity_magnitude)
    W = jnp.nan_to_num(dY * velocity_magnitude)

    # Sum contributions from all vortices
    V_total = jnp.nansum(V, axis=1)
    W_total = jnp.nansum(W, axis=1)

    return V_total, W_total


def _calculate_time_step(V_induced, W_induced, cfl_factor, dist2):
    """Calculate adaptive time step based on CFL condition.
    """
    min_distance = jnp.sqrt(jnp.min(dist2))
    max_velocity = jnp.max(jnp.abs(jnp.concatenate([V_induced, W_induced])))

    has_velocity = max_velocity > 0
    safe_velocity = jnp.where(has_velocity, max_velocity, 1.0)
    return jnp.where(has_velocity, min_distance / safe_velocity * cfl_factor, 1e-4)

_GRAVEYARD = 1.0e6  # finite (not inf, to avoid 0*inf -> nan if masking is ever bypassed),
                     # far outside any realistic domain for this model

#: Critical core-to-separation ratio a/b above which two co-rotating vortices merge.
#: None restores the legacy absolute-distance criterion (config.merge_threshold * D).
#:
#: SET TO None PENDING A VERDICT. The 0.29 criterion describes merger of physically
#: DISTINCT vortices, and it is not obvious that adjacent elements of a discretised
#: continuous sheet qualify -- with Nv = 20 on a circle of circumference pi*D the element
#: spacing is 0.157 D against cores of 0.05 D, i.e. a/b = 0.32, already at the critical
#: ratio. So the criterion would merge every adjacent pair on the first step and collapse
#: the ring, which coarsens the discretisation rather than modelling physics. Measure
#: before adopting; see the note on the result below.
MERGE_CORE_RATIO = None

#: How many nearest neighbours of each vortex the merge scan considers. Must be >= the
#: most vortices ever within the merge threshold of one vortex, or merges are silently
#: missed; MEASURED maximum is 2 over every step of four representative cases, so 12 is
#: a six-fold margin. Cost is linear in it.
MERGE_NEIGHBOURS = 12


def _merge_close_vortices(vortex_field, threshold, dist_matrix):
    """Merge vortices that are closer than the threshold distance.
    """
    Circ0_full = vortex_field.Circ
    active_full = vortex_field.active
    N_cap = Circ0_full.shape[0] // 2

    Y0 = vortex_field.Y[:N_cap]
    Z0 = vortex_field.Z[:N_cap]
    Rv0 = vortex_field.Rv[:N_cap]
    age0 = vortex_field.age[:N_cap]
    Circ0 = Circ0_full[:N_cap]
    active0 = active_full[:N_cap]

    real_dist = dist_matrix[:N_cap, :N_cap]

    # CANDIDATE PAIRS, not all of them. This scan was the single most expensive thing in
    # the model: it walked all N_cap*(N_cap-1)/2 pairs SEQUENTIALLY every time step --
    # claimed/locked make each merge depend on the ones before it, so it cannot vectorise
    # -- and that is what drove the n^2.29 farm-size scaling. MEASURED per vortex step:
    #
    #     farm   N(vtx)   pairwise   project   merge
    #     1x3       246     1.0 ms    5.0 ms    33 ms
    #     3x3       738     4.1 ms     76 ms   435 ms
    #     4x4      1312     9.1 ms    191 ms  1292 ms
    #     6x6      2952    36.6 ms   1677 ms  6840 ms      <- 80-87% of the step
    #
    # Nearly every one of those pairs is far apart and can never qualify. MEASURED over
    # every step of the full-wake optimum, the (+40,-40) sweep edge, the 8-turbine row and
    # a yawed 3x3: the most vortices ever within merge_threshold of one vortex is TWO, and
    # the median is ZERO. So scanning each vortex's MERGE_NEIGHBOURS nearest instead is
    # exact with a wide margin, and shortens the scan from N^2/2 to N*k.
    #
    # EXACTNESS, which is the only reason this is allowed to be a pure speedup: the scan
    # walks pairs in INDEX order and locks, so the partner that wins for vortex i is the
    # lowest-INDEX qualifying one, NOT the nearest. Restricting to k nearest is therefore
    # only identical if k covers every QUALIFYING neighbour -- then the candidate set
    # contains every qualifying pair, re-sorting by i*N_cap + j restores the triu order
    # the scan used to see, and the pairs dropped were no-ops. Both orientations are built
    # and folded to (min, max) because j may list i without i listing j; the resulting
    # duplicate pairs are harmless, since `locked` makes the second visit inert.
    #
    # test_merge_candidates_cover_every_qualifying_pair guards k.
    #
    # Deferred into a function so the OUTER gate below can skip building the list at all.
    # top_k is O(N_cap^2 log k) and the argsort O(N_cap k log N_cap k), and both ran every
    # step to assemble a candidate list that is almost always empty: measured at 25% of a
    # vortex step on a 10x8 farm, against 19% for the whole cloud-on-cloud induction.
    k_nbr = min(MERGE_NEIGHBOURS, N_cap - 1)

    def _candidates():
        _, nbr = lax.top_k(-real_dist, k_nbr)          # nearest first; diagonal is +inf
        src = jnp.repeat(jnp.arange(N_cap), k_nbr)
        dst = nbr.reshape(-1)
        lo = jnp.minimum(src, dst)
        hi = jnp.maximum(src, dst)
        order = jnp.argsort(lo * N_cap + hi)           # back into triu order
        i_idx, j_idx = lo[order], hi[order]
        # A degenerate row -- an inactive vortex, whose whole row of dist_matrix is +inf --
        # gives top_k an all-equal score and it may return the vortex itself. can_merge
        # would reject it anyway via active0, but excluding it here keeps `qualifies`
        # meaning what it says.
        distinct = i_idx != j_idx
        if MERGE_CORE_RATIO is None:
            qual = (real_dist[i_idx, j_idx] <= threshold**2) & distinct
        else:
            # Physical merger criterion instead of an absolute distance: two co-rotating
            # vortices merge once their core-to-separation ratio a/b exceeds ~0.29
            # (Meunier & Leweke 2001; Cerretelli & Williamson 2003; Griffiths & Hopfinger
            # 1987 give 0.29-0.32, Dritschel's 2D stability bound is 0.32). Using the
            # CURRENT core radius couples merging to the vortex viscosity the model already
            # evolves -- cores grow as a^2 = a0^2 + 4*nu*t, so the merge distance grows
            # downstream, which is the physically right behaviour and removes an absolute
            # length from the model.
            a_eff2 = 0.5 * (Rv0[i_idx] ** 2 + Rv0[j_idx] ** 2)  # RMS core; = a for equal cores
            qual = (real_dist[i_idx, j_idx] <= jnp.maximum(
                a_eff2 / MERGE_CORE_RATIO ** 2, threshold ** 2)) & distinct
        return i_idx, j_idx, qual

    # The widest separation at which ANY pair could possibly merge. Under the absolute
    # criterion that is the threshold; under the relative one the per-pair bound is
    # a_eff2 = (Rv_i^2 + Rv_j^2)/2, which cannot exceed max(Rv)^2 over the ACTIVE vortices,
    # since can_merge requires both members active. So this bounds BOTH criteria, and
    # unlike a bare threshold test it does not silently stop being a bound the day
    # MERGE_CORE_RATIO is switched on.
    if MERGE_CORE_RATIO is None:
        r_max2 = threshold ** 2
    else:
        r_max2 = jnp.maximum(
            jnp.max(jnp.where(active0, Rv0, 0.0)) ** 2 / MERGE_CORE_RATIO ** 2,
            threshold ** 2)

    # NOT compacted to the qualifying pairs, though it is tempting: 98-99.9% of steps
    # have none at all, and gathering them into a fixed buffer made merge 1.8x faster
    # again. TRIED AND REVERTED -- it is not lossless. Sizing the buffer from the
    # qualifying-pair count measured per turbine (max 48) overran it, because
    # solve_single_field CONCATENATES the rotors' clouds, so the live N_cap during a farm
    # march is far larger than any single turbine's stored field suggests, and so is the
    # pair count. Two of six cases moved (full-wake C in the 7th digit, the 8-turbine row
    # in the 5th) from merges being dropped once the buffer filled.
    #
    # A buffer can only be made safe by a bound that holds for the CONCATENATED cloud,
    # and jit cannot raise when one is exceeded -- so the failure would stay silent.
    # Measure the pair count inside the march, not per turbine, before trying again.

    def merge_step(carry, xs):
        Y, Z, Rv, age, Circ, claimed, locked = carry
        i, j, qualifies = xs
        can_merge = qualifies & active0[i] & active0[j] & (~locked[i]) & (~locked[j])

        Circ_i, Circ_j = Circ[i], Circ[j]
        total_circ = Circ_i + Circ_j
        combine = jnp.abs(total_circ) > 1e-12
        denom = jnp.where(combine, total_circ, 1.0)
        merged_Y = (Y[i] * Circ_i + Y[j] * Circ_j) / denom
        merged_Z = (Z[i] * Circ_i + Z[j] * Circ_j) / denom
        # CLAMPED BETWEEN THE PARENTS. The circulation centroid divides by Gi + Gj, so a
        # NEARLY cancelling pair -- which passes the 1e-12 test above -- places the
        # survivor at a distance amplified by |Gi| / |Gi + Gj|, still flagged active.
        #
        # This is not hypothetical once the cloud holds rings from rotors at OPPOSITE yaw,
        # which _concat_vortices splices together and a layout optimiser is free to
        # propose. Measured as the smallest |Gi+Gj| / (|Gi|+|Gj|) over pairs that actually
        # qualify for merging:
        #
        #     (0,0,0)       1.000     no cancellation; the ring is single-signed
        #     (+25,+15,0)   0.0984
        #     (+30,-30,0)   0.0400
        #     (+40,-40,0)   0.0169    -> ~59x amplification
        #
        # and +-40 deg rows are IN the 117-point yaw sweep. At 0.0169 with a separation of
        # merge_threshold = 0.05 D the survivor lands ~1.5 D from either parent.
        #
        # A relative annihilation tolerance does not bound this: at 1e-3 it would still
        # admit ~500x, i.e. 25 D. Clamping does, with no tolerance to choose -- a merged
        # vortex outside the segment joining its parents is unphysical whatever the
        # circulations, and for a cancelling pair the centroid is meaningless anyway.
        # Same-signed pairs are a convex combination already, so the clamp is inert there
        # and this cannot change a single-signed cloud.
        merged_Y = jnp.clip(merged_Y, jnp.minimum(Y[i], Y[j]), jnp.maximum(Y[i], Y[j]))
        merged_Z = jnp.clip(merged_Z, jnp.minimum(Z[i], Z[j]), jnp.maximum(Z[i], Z[j]))

        w_i, w_j = jnp.abs(Circ_i), jnp.abs(Circ_j)
        w_sum = jnp.where(w_i + w_j > 1e-12, w_i + w_j, 1.0)
        merged_Rv = jnp.sqrt((w_i * Rv[i]**2 + w_j * Rv[j]**2) / w_sum)
        merged_age = (w_i * age[i] + w_j * age[j]) / w_sum

        do_combine = can_merge & combine
        do_remove_both = can_merge & (~combine)

        Y = Y.at[j].set(jnp.where(do_combine, merged_Y, Y[j]))
        Z = Z.at[j].set(jnp.where(do_combine, merged_Z, Z[j]))
        Rv = Rv.at[j].set(jnp.where(do_combine, merged_Rv, Rv[j]))
        age = age.at[j].set(jnp.where(do_combine, merged_age, age[j]))
        # The survivor must CARRY the absorbed vortex's circulation: Zong Sec 4.2 combines
        # two vortices into one. Previously only position/radius/Nu were merged while Circ
        # kept its own value and the absorbed Gamma was deleted, so EVERY merge destroyed
        # circulation -- even a same-signed pair, which must conserve it exactly. That made
        # the loss track the merge count and hence Nv, which is why Zong's reported
        # insensitivity above N=20 could not be reproduced.
        Circ = Circ.at[j].set(jnp.where(do_combine, total_circ, Circ[j]))
        claimed = claimed.at[i].set(claimed[i] | can_merge)       # i is absorbed
        claimed = claimed.at[j].set(claimed[j] | do_remove_both)  # j only if they cancelled
        locked = locked.at[i].set(locked[i] | can_merge)
        locked = locked.at[j].set(locked[j] | can_merge)          # j is done for this step
        return (Y, Z, Rv, age, Circ, claimed, locked), None

    _none = jnp.zeros(N_cap, dtype=bool)
    init_carry = (Y0, Z0, Rv0, age0, Circ0, _none, _none)

    # SKIP THE WHOLE MERGE WHEN NOTHING CAN QUALIFY, which the note above measures at
    # 98-99.9% of steps. Two gates, because the two halves cost differently.
    #
    # OUTER, on min(real_dist): skips building the candidate list as well as walking it.
    # The list costs a top_k over the N_cap^2 matrix plus an argsort, which together were
    # 25% of a vortex step on a 10x8 farm -- more than the cloud-on-cloud induction. A
    # single O(N_cap^2) min with no sort replaces them whenever the closest pair in the
    # whole cloud is wider apart than any merge radius could be.
    #
    # INNER, on any(qualifies_all): the scan is sequential by construction -- claimed and
    # locked make each pair depend on the ones before it -- so an empty one still costs
    # N_cap*k dependent iterations to compute nothing. Under the absolute criterion the
    # outer gate already implies this one (the closest pair is in its own k-nearest), but
    # under the relative criterion the outer bound uses max(Rv) and can pass when no
    # individual pair qualifies, so the inner gate still earns its place. It is an any()
    # over an array already built.
    #
    # Both are exact, not approximate. With no qualifying pair every can_merge is False,
    # so each `.at[j].set(where(False, new, old))` writes the old value back and
    # claimed/locked stay all-False: the scan IS the identity on the carry, which is the
    # other branch. And r_max2 bounds both merge criteria, so neither gate can hide a
    # merge the scan would have made.
    def _merge_pass(carry):
        i_idx, j_idx, qualifies_all = _candidates()

        def _scan_merges(c):
            out, _ = lax.scan(merge_step, c, (i_idx, j_idx, qualifies_all))
            return out

        return lax.cond(jnp.any(qualifies_all), _scan_merges, lambda c: c, carry)

    (Y, Z, Rv, age, Circ, claimed, _locked) = lax.cond(
        jnp.min(real_dist) <= r_max2, _merge_pass, lambda carry: carry, init_carry)

    active_new = active0 & (~claimed)
    Circ_new = jnp.where(claimed, 0.0, Circ)
    Y_new = jnp.where(claimed, _GRAVEYARD, Y)
    Z_new = jnp.where(claimed, _GRAVEYARD, Z)

    return VortexField(
        Y=jnp.concatenate([Y_new, Y_new]),
        Z=jnp.concatenate([Z_new, -Z_new]),
        Rv=jnp.concatenate([Rv, Rv]),
        age=jnp.concatenate([age, age]),
        Circ=jnp.concatenate([Circ_new, -Circ_new]),
        active=jnp.concatenate([active_new, active_new]),
        yloc=vortex_field.yloc,
        zloc=vortex_field.zloc,
        V=vortex_field.V,
        W=vortex_field.W,
        OmegaX=vortex_field.OmegaX,
        t=vortex_field.t
    )

def _fast_dct2(x):
    """DCT-II along the last axis, Makhoul-style on a real FFT.

    jax.scipy.fft.dct is written on a full COMPLEX fft plus a pad and a reorder, which
    makes a DST-II of length H cost MORE than the rfft of length 2H it is meant to
    replace (2.20 ms against 0.736 at farm scale). jax has no dst at all.
    """
    H = x.shape[-1]
    v = jnp.concatenate([x[..., 0::2], x[..., 1::2][..., ::-1]], axis=-1)
    u = jnp.fft.rfft(v, axis=-1) * jnp.exp(-1j * jnp.pi * jnp.arange(H // 2 + 1) / (2 * H))
    # the two branches overlap at k = H/2 and agree there because rfft(v)[H/2] is real
    return jnp.concatenate([2 * jnp.real(u), -2 * jnp.imag(u[..., H // 2 - 1:0:-1])], -1)


def _fast_dst2(x):
    return _fast_dct2(x * (-1.0) ** jnp.arange(x.shape[-1]))[..., ::-1]


def _idst2(S):
    n = S.shape[-1]
    return jax.scipy.fft.idct(S[..., ::-1], type=2, axis=-1) * (-1.0) ** jnp.arange(n)


def pm_ladder(D, tol, lo_D=0.04, hi_D=2.2):
    """The fixed geometric ladder of core radii the PM solver bins onto.

    It must span every Rv the whole march produces, not the spread of any one step: Rv
    grows as Rv^2 += 4 nu dt and merging makes new cores, so a snapshot spans ~1.5x while
    a full run spans 23x (3-rotor case) to 34x (8-rotor row). Sizing the ladder from a
    snapshot would silently clamp most of the run into the end rungs.
    """
    n = int(np.ceil(np.log(hi_D / lo_D) / np.log1p(tol)))
    return lo_D * D * (1.0 + tol) ** (np.arange(n) + 0.5), lo_D * D, float(np.log1p(tol))


def pm_kernels(cores, yloc, zloc):
    """Av, Aw per ladder rung: the whole cost of the ground image, paid once per run.

    psi = 0 on the ground, so the field is ODD in z and the mirror vortices carry no
    information. Solving on the half domain imposes that by construction, which halves
    both the grid and the vortex count. The grid is CELL-CENTRED, so the ground is a
    HALF-SAMPLE point and this is the DST-II family, not DST-I.

    With Kv ~ -z f(r2) odd in z and Kw ~ y f(r2) even, the phase factors of the odd
    extension cancel and the solve is real arithmetic on half-length transforms:

        dct2(V)[k]   = -i Kv_hat[k] dst2(src)[k-1]        V is EVEN in z
        dst2(W)[k-1] =    Kw_hat[k] dst2(src)[k-1]        W is ODD  in z

    The odd kernel's NYQUIST row must be zeroed: offsets -P/2 and +P/2 are the same
    lattice point, so an odd kernel is only self-consistent at zero there, and leaving it
    makes Kv_hat complex and corrupts the cosine branch by ~5e-4. That is exact only
    because H = 2 nz keeps every source-target pair inside |offset| < P/2 -- the largest
    is 2 nz - 1, which is why the half array is twice the physical grid and not less.
    """
    yl, zl = np.asarray(yloc), np.asarray(zloc)
    ny, nz = yl.shape
    dy, dz = yl[1, 0] - yl[0, 0], zl[0, 1] - zl[0, 0]
    Py, H = 2 * ny, 2 * nz
    P = 2 * H
    ay, aq = np.arange(Py), np.arange(P)
    MY = np.where(ay < Py // 2, ay, ay - Py)[:, None] * dy
    MZ = np.where(aq < P // 2, aq, aq - P)[None, :] * dz
    R2 = MY ** 2 + MZ ** 2
    R2s = np.where(R2 == 0, 1.0, R2)
    Av = np.empty((len(cores), Py // 2 + 1, H - 1), np.complex128)
    Aw = np.empty((len(cores), Py // 2 + 1, H), np.complex128)
    for i, a in enumerate(cores):
        p = np.where(R2 == 0, 0.0, (1 - np.exp(-R2 / a ** 2)) / (2 * np.pi * R2s))
        Kv, Kw = -p * MZ, p * MY
        Kv[:, P // 2] = 0.0
        Av[i] = -1j * np.fft.fft(np.fft.rfft(Kv, axis=0), axis=1)[:, 1:H]
        Aw[i] = np.fft.fft(np.fft.rfft(Kw, axis=0), axis=1)[:, 1:H + 1]
    return jnp.asarray(Av), jnp.asarray(Aw)


def _oseenlamb_pm(Circ, Y, Z, Rv, yloc, zloc, Av, Aw, lo, lnr):
    """Particle-mesh form of _oseenlamb: same field, O(bins * M log M) instead of O(N*M).

    Only the REAL half of the cloud is deposited; the mirrors are reproduced by the odd
    symmetry of the basis. That is safe because the march keeps the halves exactly
    paired -- verified at 0.00e+00 over 14000 steps -- and the pairing is relied on here,
    so it is asserted in the invariant tests rather than assumed.
    """
    ny, nz = yloc.shape
    Py, H = 2 * ny, 2 * nz
    oy = (Py - ny) // 2
    dy = yloc[1, 0] - yloc[0, 0]
    dz = zloc[0, 1] - zloc[0, 0]
    nb = Av.shape[0]
    N = Circ.shape[0] // 2

    c0, Y0, Z0, R0 = Circ[:N], Y[:N], Z[:N], Rv[:N]
    gid = jnp.clip((jnp.log(jnp.maximum(R0, lo) / lo) / lnr).astype(int), 0, nb - 1)

    gy = (Y0 - (yloc[0, 0] - oy * dy)) / dy
    gz = Z0 / dz - 0.5                       # cell j is centred at (j + 1/2) dz
    # Retired vortices are parked far off-grid and carry no circulation; clamp so their
    # index arithmetic cannot overflow or alias onto a live cell.
    gy = jnp.clip(jnp.nan_to_num(gy), -1.0, Py - 2.0)
    gz = jnp.clip(jnp.nan_to_num(gz), -0.5, H - 2.0)
    i0, j0 = jnp.floor(gy).astype(int), jnp.floor(gz).astype(int)
    fy, fz = gy - i0, gz - j0

    # The bins are a BATCH AXIS, not a loop. Unrolling them in Python cost ~15 ops per
    # bin and ~600 kernel launches per projection, and at real farm grid sizes that
    # overhead beat the transforms it was meant to save -- it is what made the landed
    # solver slower than the direct sum below ~16 rotors.
    #
    # Scattering into (nb, Py, H) indexed by gid also drops the bin factor out of the
    # DEPOSIT: every vortex is written once, to its own bin's slice, instead of all N
    # being masked and re-scanned for each of the nb bins.
    src = jnp.zeros((nb, Py, H))
    for di, wy in ((0, 1 - fy), (1, fy)):
        for dj, wz in ((0, 1 - fz), (1, fz)):
            jj = j0 + dj
            # Weight landing at index -1 belongs to this vortex's OWN mirror cell, so it
            # folds back to index 0 with a SIGN FLIP. Letting it wrap to the top of the
            # array instead looks harmless and is not.
            jf = jnp.where(jj < 0, -1 - jj, jj)
            sg = jnp.where(jj < 0, -1.0, 1.0)
            src = src.at[gid, (i0 + di) % Py, jf % H].add(c0 * wy * wz * sg)

    Sh = jnp.fft.rfft(_fast_dst2(src), axis=1)            # (nb, Py//2+1, H)
    Cvh = jnp.zeros((Py // 2 + 1, H), Sh.dtype)
    Cvh = Cvh.at[:, 1:].set(jnp.sum(Av * Sh[:, :, :H - 1], axis=0))
    Swh = jnp.sum(Aw * Sh, axis=0)

    V = jax.scipy.fft.idct(jnp.fft.irfft(Cvh, n=Py, axis=0), type=2, axis=-1)
    W = _idst2(jnp.fft.irfft(Swh, n=Py, axis=0))
    return V[oy:oy + ny, :nz], W[oy:oy + ny, :nz]


def _oseenlamb(Circ, Y, Z, Rv, yloc, zloc):
    # Reshape vortex properties to (N, 1, 1) to enable broadcasting against (H, W)
    Y = Y[:, jnp.newaxis, jnp.newaxis]
    Z = Z[:, jnp.newaxis, jnp.newaxis]
    Circ = Circ[:, jnp.newaxis, jnp.newaxis]
    Rv = Rv[:, jnp.newaxis, jnp.newaxis]

    # Calculate relative distances: Shape (N, H, W)
    dY = yloc[jnp.newaxis, :, :] - Y
    dZ = zloc[jnp.newaxis, :, :] - Z
    r2 = dY**2 + dZ**2

    r2_safe = jnp.maximum(r2, 1e-8)  # Prevent division by zero
    prefactor = (Circ / (2 * jnp.pi * r2_safe)) * (1 - jnp.exp(-r2 / Rv**2))

    V = jnp.sum(-prefactor * dZ, axis=0)
    W = jnp.sum(prefactor * dY, axis=0)
    return V, W

def RK4_step(Y, Z, V_k1, W_k1, Circ, Rv, dt, active=None):
    """Integrates positions using RK4 and enforces ground symmetry.
    """
    N_real = len(Circ) // 2

    # k2: Midpoint using k1
    Y2 = Y + V_k1 * (dt/2)
    Z2 = Z + W_k1 * (dt/2)
    Z2 = Z2.at[N_real:].set(-Z2[:N_real]) # Enforce ground symmetry
    k2_v, k2_w = _get_vortex_derivatives(Y2, Z2, Circ, Rv, active)

    # k3: Midpoint using k2
    Y3 = Y + k2_v * (dt/2)
    Z3 = Z + k2_w * (dt/2)
    Z3 = Z3.at[N_real:].set(-Z3[:N_real]) # Enforce ground symmetry
    k3_v, k3_w = _get_vortex_derivatives(Y3, Z3, Circ, Rv, active)

    # k4: End point using k3
    Y4 = Y + k3_v * dt
    Z4 = Z + k3_w * dt
    Z4 = Z4.at[N_real:].set(-Z4[:N_real]) # Enforce ground symmetry
    k4_v, k4_w = _get_vortex_derivatives(Y4, Z4, Circ, Rv, active)

    # Final Position Update
    new_Y = Y + (dt/6.0) * (V_k1 + 2*k2_v + 2*k3_v + k4_v)
    new_Z = Z + (dt/6.0) * (W_k1 + 2*k2_w + 2*k3_w + k4_w)
    new_Z = new_Z.at[N_real:].set(-new_Z[:N_real]) # Final symmetry enforcement

    return new_Y, new_Z

def _get_relative_geometry(Y, Z, active=None):
    """Compute relative geometry matrices for vortices.
    """
    # Create distance matrices
    Y_grid, Y_grid_T = jnp.meshgrid(Y, Y, indexing='ij')
    Z_grid, Z_grid_T = jnp.meshgrid(Z, Z, indexing='ij')

    dY = Y_grid - Y_grid_T
    dZ = Z_grid - Z_grid_T
    dist2 = dY**2 + dZ**2

    # Fill diagonal with inf to avoid division by zero
    n = Y.shape[0]
    dist2 = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist2)

    if active is not None:
        inactive = ~active
        dist2 = jnp.where(inactive[:, jnp.newaxis], jnp.inf, dist2)
        dist2 = jnp.where(inactive[jnp.newaxis, :], jnp.inf, dist2)

    return dY, dZ, dist2

def _get_vortex_derivatives(Y, Z, Circ, Rv, active=None):
    """Calculates the velocity of each vortex core."""
    dY, dZ, dist2 = _get_relative_geometry(Y, Z, active)

    V_ind, W_ind = _compute_mutual_induction(Circ, Rv, dY, dZ, dist2)
    return V_ind, W_ind
