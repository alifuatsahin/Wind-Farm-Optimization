import dataclasses
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .data_structures import VortexField

_UNSTACKED = jnp.zeros((0,))


@partial(jax.jit, static_argnames=("total_steps",))
def _simulate_vortex_evolution_jit(config, merge_threshold, cfl_factor, total_steps,
                                   seed=None):
    D = config.D
    Uhub = config.Uhub
    t_limit = 2.5 * config.calculation_domain / D

    if seed is None:
        seed = _define_location(config)[0]
    seed = dataclasses.replace(seed, yloc=config.yloc, zloc=config.zloc, OmegaX=jnp.zeros_like(config.yloc))

    def step(current_state, t):
        current2 = _vortex2velocity(current_state, config)
        dY, dZ, dist2 = _get_relative_geometry(current2.Y, current2.Z, current2.active)
        Circ, Rv = current2.Circ, current2.Rv
        V_induced, W_induced = _compute_mutual_induction(Circ, Rv, dY, dZ, dist2)
        dt = _calculate_time_step(V_induced, W_induced, cfl_factor, dist2)
        t_next = t + dt
        would_stop = t_next * Uhub / D > t_limit

        def do_advance(_):
            new_Y, new_Z = RK4_step(current2.Y, current2.Z, V_induced, W_induced, Circ, Rv, dt, current2.active)
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
            return merged, t_next, True

        def do_freeze(_):
            return current2, t_next, False

        return lax.cond(would_stop, do_freeze, do_advance, operand=None), current2

    def unstack(frame):
        return dataclasses.replace(frame, yloc=_UNSTACKED, zloc=_UNSTACKED, U=_UNSTACKED)

    def advance(carry):
        current_state, t, _ = carry
        next_carry, emitted = step(current_state, t)
        return next_carry, unstack(emitted)

    stacked, was_active, _ = march_until_stopped(
        advance, (seed, 0.0, True), total_steps, idle_frame=lambda carry: unstack(carry[0]))
    stacked = dataclasses.replace(stacked, yloc=config.yloc, zloc=config.zloc,
                                  U=_UNSTACKED)
    return stacked, was_active


def march_until_stopped(advance, init, total_steps, idle_frame=None):
    """A `total_steps`-long march that stops computing once it reports itself inactive.

    `advance(carry) -> (next_carry, frame)`, with the carry's LAST member the active flag.
    Returns the frames stacked along a leading axis of length `total_steps`, the per-step
    active flags and the final carry -- what a lax.scan of `total_steps` gives when its inactive steps emit
    `idle_frame(carry)` and pass the carry through, but without running those idle steps: the
    march uses as few as 4% of its allotted steps. `idle_frame` fills the unused tail, which
    matters to a reader of the whole stack (interpolate_vec_data searchsorts its t, so a zero
    tail would be read as frames at t=0); with None the tail is left as zeros, for a caller
    that only ever reads the active prefix.
    """
    frame_aval = jax.eval_shape(lambda c: advance(c)[1], init)
    frames = jax.tree.map(lambda a: jnp.zeros((total_steps,) + a.shape, a.dtype), frame_aval)

    def body(loop):
        i, carry, frames, was_active = loop
        # The loop has already stopped by the time the flag is False, so this cond never
        # takes its idle side. It is kept because it is what the scan this replaced compiled:
        # on the CPU backend the same step compiled outside a conditional sums the PM
        # scatter in a different order (objective ~1e-8 off on a 16-rotor farm), and inside
        # one the results stay bit-identical to the scan's.
        carry, frame = lax.cond(carry[-1], advance,
                                lambda c: (c, jax.tree.map(jnp.zeros_like, frame_aval)), carry)
        frames = jax.tree.map(lambda buf, x: buf.at[i].set(x), frames, frame)
        return i + 1, carry, frames, was_active.at[i].set(True)

    n, carry, frames, was_active = lax.while_loop(
        lambda loop: loop[1][-1] & (loop[0] < total_steps), body,
        (0, init, frames, jnp.zeros(total_steps, dtype=bool)))
    if idle_frame is not None:
        tail = jnp.arange(total_steps) >= n
        frames = jax.tree.map(
            lambda buf, x: jnp.where(tail.reshape((-1,) + (1,) * jnp.ndim(x)), x, buf),
            frames, idle_frame(carry))
    return frames, was_active, carry

def _vortex2velocity(data, config):
    """Returns a NEW VortexField (dataclasses.replace) instead of mutating `data` in place..."""
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
    Y = config.D / 2 * jnp.cos(config.phi) * jnp.cos(config.beta) - config.Yoffset
    Z = config.D / 2 * jnp.sin(config.phi) + config.Zhub
    Rv = jnp.full(config.Nv, 0.05 * config.D)
    Circ = jnp.asarray(config.dgamma)

    Y = jnp.append(Y, -config.Yoffset)
    Z = jnp.append(Z, config.Zhub)
    Rv = jnp.append(Rv, 0.15 * config.D)
    Circ = jnp.append(Circ, -jnp.sum(config.dgamma))

    Y = jnp.concatenate([Y, Y])
    Z = jnp.concatenate([Z, -Z])
    Rv = jnp.concatenate([Rv, Rv])
    Circ = jnp.concatenate([Circ, -Circ])
    active = jnp.ones_like(Y, dtype=bool)

    age = jnp.zeros_like(Y)

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
    return [vordata]

def _compute_mutual_induction(Circ, Rv, dY, dZ, dist2):
    """Compute induced velocities using Biot-Savart law with core correction."""
    velocity_magnitude = (Circ[jnp.newaxis, :] / (2 * jnp.pi * dist2) *
                        (1 - jnp.exp(-dist2 / (Rv[jnp.newaxis, :]**2))))
    velocity_magnitude = jnp.nan_to_num(velocity_magnitude)

    V = jnp.nan_to_num(-dZ * velocity_magnitude)
    W = jnp.nan_to_num(dY * velocity_magnitude)

    V_total = jnp.nansum(V, axis=1)
    W_total = jnp.nansum(W, axis=1)

    return V_total, W_total


def _calculate_time_step(V_induced, W_induced, cfl_factor, dist2):
    """Calculate adaptive time step based on CFL condition."""
    min_distance = jnp.sqrt(jnp.min(dist2))
    max_velocity = jnp.max(jnp.abs(jnp.concatenate([V_induced, W_induced])))

    has_velocity = max_velocity > 0
    safe_velocity = jnp.where(has_velocity, max_velocity, 1.0)
    return jnp.where(has_velocity, min_distance / safe_velocity * cfl_factor, 1e-4)

_GRAVEYARD = 1.0e6

MERGE_CORE_RATIO = None

MERGE_NEIGHBOURS = 12


def _merge_close_vortices(vortex_field, threshold, dist_matrix):
    """Merge vortices that are closer than the threshold distance."""
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

    k_nbr = min(MERGE_NEIGHBOURS, N_cap - 1)

    def _candidates():
        _, nbr = lax.top_k(-real_dist, k_nbr)
        src = jnp.repeat(jnp.arange(N_cap), k_nbr)
        dst = nbr.reshape(-1)
        lo = jnp.minimum(src, dst)
        hi = jnp.maximum(src, dst)
        order = jnp.argsort(lo * N_cap + hi)
        i_idx, j_idx = lo[order], hi[order]
        distinct = i_idx != j_idx
        if MERGE_CORE_RATIO is None:
            qual = (real_dist[i_idx, j_idx] <= threshold**2) & distinct
        else:
            a_eff2 = 0.5 * (Rv0[i_idx] ** 2 + Rv0[j_idx] ** 2)
            qual = (real_dist[i_idx, j_idx] <= jnp.maximum(
                a_eff2 / MERGE_CORE_RATIO ** 2, threshold ** 2)) & distinct
        return i_idx, j_idx, qual

    if MERGE_CORE_RATIO is None:
        r_max2 = threshold ** 2
    else:
        r_max2 = jnp.maximum(
            jnp.max(jnp.where(active0, Rv0, 0.0)) ** 2 / MERGE_CORE_RATIO ** 2,
            threshold ** 2)


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
        Circ = Circ.at[j].set(jnp.where(do_combine, total_circ, Circ[j]))
        claimed = claimed.at[i].set(claimed[i] | can_merge)
        claimed = claimed.at[j].set(claimed[j] | do_remove_both)
        locked = locked.at[i].set(locked[i] | can_merge)
        locked = locked.at[j].set(locked[j] | can_merge)
        return (Y, Z, Rv, age, Circ, claimed, locked), None

    _none = jnp.zeros(N_cap, dtype=bool)
    init_carry = (Y0, Z0, Rv0, age0, Circ0, _none, _none)

    def _merge_pass(carry):
        i_idx, j_idx, qualifies_all = _candidates()
        # merge_step is an exact no-op on a pair that does not qualify, so walk only the ones
        # that do, in the same order. The candidate list is 12 pairs per vortex and the walk
        # is sequential: over all of it, this pass was ~99% of a step for an 80-rotor cloud.
        order = jnp.argsort((~qualifies_all).astype(jnp.int32), stable=True)
        i_q, j_q = i_idx[order], j_idx[order]
        return lax.fori_loop(0, jnp.sum(qualifies_all),
                             lambda k, c: merge_step(c, (i_q[k], j_q[k], True))[0], carry)

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
    """DCT-II along the last axis, Makhoul-style on a real FFT."""
    H = x.shape[-1]
    v = jnp.concatenate([x[..., 0::2], x[..., 1::2][..., ::-1]], axis=-1)
    u = jnp.fft.rfft(v, axis=-1) * jnp.exp(-1j * jnp.pi * jnp.arange(H // 2 + 1) / (2 * H))
    return jnp.concatenate([2 * jnp.real(u), -2 * jnp.imag(u[..., H // 2 - 1:0:-1])], -1)


def _fast_dst2(x):
    return _fast_dct2(x * (-1.0) ** jnp.arange(x.shape[-1]))[..., ::-1]


def _idst2(S):
    n = S.shape[-1]
    return jax.scipy.fft.idct(S[..., ::-1], type=2, axis=-1) * (-1.0) ** jnp.arange(n)


def pm_ladder(D, tol, lo_D=0.04, hi_D=2.2):
    """The fixed geometric ladder of core radii the PM solver bins onto."""
    n = int(np.ceil(np.log(hi_D / lo_D) / np.log1p(tol)))
    return lo_D * D * (1.0 + tol) ** (np.arange(n) + 0.5), lo_D * D, float(np.log1p(tol))


def pm_kernels(cores, yloc, zloc):
    """Av, Aw per ladder rung: the whole cost of the ground image, paid once per run."""
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
    """Particle-mesh form of _oseenlamb: same field, O(bins * M log M) instead of O(N*M)."""
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
    gz = Z0 / dz - 0.5
    gy = jnp.clip(jnp.nan_to_num(gy), -1.0, Py - 2.0)
    gz = jnp.clip(jnp.nan_to_num(gz), -0.5, H - 2.0)
    i0, j0 = jnp.floor(gy).astype(int), jnp.floor(gz).astype(int)
    fy, fz = gy - i0, gz - j0

    src = jnp.zeros((nb, Py, H))
    for di, wy in ((0, 1 - fy), (1, fy)):
        for dj, wz in ((0, 1 - fz), (1, fz)):
            jj = j0 + dj
            jf = jnp.where(jj < 0, -1 - jj, jj)
            sg = jnp.where(jj < 0, -1.0, 1.0)
            src = src.at[gid, (i0 + di) % Py, jf % H].add(c0 * wy * wz * sg)

    Sh = jnp.fft.rfft(_fast_dst2(src), axis=1)
    Cvh = jnp.zeros((Py // 2 + 1, H), Sh.dtype)
    Cvh = Cvh.at[:, 1:].set(jnp.sum(Av * Sh[:, :, :H - 1], axis=0))
    Swh = jnp.sum(Aw * Sh, axis=0)

    V = jax.scipy.fft.idct(jnp.fft.irfft(Cvh, n=Py, axis=0), type=2, axis=-1)
    W = _idst2(jnp.fft.irfft(Swh, n=Py, axis=0))
    return V[oy:oy + ny, :nz], W[oy:oy + ny, :nz]


def _oseenlamb(Circ, Y, Z, Rv, yloc, zloc):
    Y = Y[:, jnp.newaxis, jnp.newaxis]
    Z = Z[:, jnp.newaxis, jnp.newaxis]
    Circ = Circ[:, jnp.newaxis, jnp.newaxis]
    Rv = Rv[:, jnp.newaxis, jnp.newaxis]

    dY = yloc[jnp.newaxis, :, :] - Y
    dZ = zloc[jnp.newaxis, :, :] - Z
    r2 = dY**2 + dZ**2

    r2_safe = jnp.maximum(r2, 1e-8)
    prefactor = (Circ / (2 * jnp.pi * r2_safe)) * (1 - jnp.exp(-r2 / Rv**2))

    V = jnp.sum(-prefactor * dZ, axis=0)
    W = jnp.sum(prefactor * dY, axis=0)
    return V, W

def RK4_step(Y, Z, V_k1, W_k1, Circ, Rv, dt, active=None):
    """Integrates positions using RK4 and enforces ground symmetry."""
    N_real = len(Circ) // 2

    Y2 = Y + V_k1 * (dt/2)
    Z2 = Z + W_k1 * (dt/2)
    Z2 = Z2.at[N_real:].set(-Z2[:N_real])
    k2_v, k2_w = _get_vortex_derivatives(Y2, Z2, Circ, Rv, active)

    Y3 = Y + k2_v * (dt/2)
    Z3 = Z + k2_w * (dt/2)
    Z3 = Z3.at[N_real:].set(-Z3[:N_real])
    k3_v, k3_w = _get_vortex_derivatives(Y3, Z3, Circ, Rv, active)

    Y4 = Y + k3_v * dt
    Z4 = Z + k3_w * dt
    Z4 = Z4.at[N_real:].set(-Z4[:N_real])
    k4_v, k4_w = _get_vortex_derivatives(Y4, Z4, Circ, Rv, active)

    new_Y = Y + (dt/6.0) * (V_k1 + 2*k2_v + 2*k3_v + k4_v)
    new_Z = Z + (dt/6.0) * (W_k1 + 2*k2_w + 2*k3_w + k4_w)
    new_Z = new_Z.at[N_real:].set(-new_Z[:N_real])

    return new_Y, new_Z

def _get_relative_geometry(Y, Z, active=None):
    """Compute relative geometry matrices for vortices."""
    Y_grid, Y_grid_T = jnp.meshgrid(Y, Y, indexing='ij')
    Z_grid, Z_grid_T = jnp.meshgrid(Z, Z, indexing='ij')

    dY = Y_grid - Y_grid_T
    dZ = Z_grid - Z_grid_T
    dist2 = dY**2 + dZ**2

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
