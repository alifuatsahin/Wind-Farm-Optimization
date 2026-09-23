import dataclasses
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .data_structures import VortexField

def simulate_vortex_evolution(config, field_params, total_steps=1000, seed=None):
    """
    Simulate mutual induction of vortices in 2D plane with adaptive time stepping.

    Args:
        config: Simulation configuration
        field_params: Field parameters
        total_steps: Maximum (and, since this always runs a fixed trip count) number of time
            steps taken; the returned list is trimmed to however many of those were naturally
            reached before the domain-exit condition fired.

    Returns:
        list[VortexField] of evolved vortex states, length = however many steps were naturally
        reached (see `_simulate_vortex_evolution_jit`'s docstring for the exact semantics).
    """
    stacked, was_active = _simulate_vortex_evolution_jit(
        config, field_params.NuT_max, field_params.merge_threshold, field_params.cfl_factor,
        total_steps, seed,
    )
    kept_count = int(jnp.sum(was_active))  # concrete Python int -- forces one sync point here,
                                            # outside jit, exactly where the plan calls for it
    return [jax.tree_util.tree_map(lambda leaf, i=i: leaf[i], stacked) for i in range(kept_count)]


@partial(jax.jit, static_argnames=("total_steps",))
def _simulate_vortex_evolution_jit(config, NuT_max, merge_threshold, cfl_factor, total_steps,
                                   seed=None):
    D = config.D
    Uhub = config.Uhub
    t_limit = 1.5 * config.calculation_domain / D  # dimensionless domain-exit threshold

    # seed=None -> this rotor sheds a fresh ring (the normal path). An explicit seed lets a
    # caller continue an existing vortex system, which is what the single-field march needs:
    # turbine i+1's vortices are appended to turbine i's still-evolving cloud rather than
    # starting a new one. See experiments/single_field_march.py.
    if seed is None:
        seed = _define_location(config, NuT_max)[0]
    seed = dataclasses.replace(seed, yloc=config.yloc, zloc=config.zloc, OmegaX=jnp.zeros_like(config.yloc))

    def scan_step(carry, _):
        current_state, t, active = carry

        def when_active(_):
            # Backfill: compute the induced-velocity field for THIS step's own (carried)
            # position -- this is what gets emitted as this step's real output.
            current2 = _vortex2velocity(current_state, config)
            dY, dZ, dist2 = _get_relative_geometry(current2.Y, current2.Z, current2.active)
            Circ, Rv = current2.Circ, current2.Rv
            V_induced, W_induced = _compute_mutual_induction(Circ, Rv, dY, dZ, dist2)
            dt = _calculate_time_step(V_induced, W_induced, cfl_factor, dist2)
            t_next = t + dt
            would_stop = t_next * Uhub / D > t_limit

            def do_advance(_):
                new_Y, new_Z = RK4_step(current2.Y, current2.Z, V_induced, W_induced, Circ, Rv, dt, current2.active)
                new_Rv = jnp.sqrt(Rv**2 + 4 * current2.Nu * dt)
                new_vf = VortexField(
                    Y=new_Y, Z=new_Z, Rv=new_Rv,
                    Circ=current2.Circ, Nu=current2.Nu, active=current2.active, t=t_next,
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
        return next_carry, (emitted, active)

    init_carry = (seed, 0.0, True)
    _, (stacked, was_active) = lax.scan(scan_step, init_carry, xs=None, length=total_steps)
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

    V, W = _oseenlamb(Circ, Y, Z, Rv, yloc, zloc)
    V = V + data.V
    W = W + data.W

    _, dVdZ = jnp.gradient(V, yloc[:, 0], zloc[0, :])
    dWdY, _ = jnp.gradient(W, yloc[:, 0], zloc[0, :])
    OmegaX = dWdY - dVdZ

    return dataclasses.replace(data, OmegaX=OmegaX, yloc=yloc, zloc=zloc, V=V, W=W)

def _define_location(config, NuT_max):
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

    # 4. Core-growth viscosity
    Nu = jnp.full(Y.shape, _calculate_viscosity(config.dgamma, NuT_max))

    # 5. Create VortexField object
    vordata = VortexField(
        Y=Y,
        Z=Z,
        Rv=Rv,
        Circ=Circ,
        Nu=Nu,
        active=active,
        yloc=jnp.array([]),
        zloc=jnp.array([]),
        V=config.V,
        W=config.W,
        OmegaX=jnp.array([]),
        t=0.0
    )
    return [vordata] # return a list of vordata for extensibility

def _calculate_viscosity(dgamma, NuT_max):
    """Cross-stream eddy viscosity governing the Lamb-Oseen core growth.

    Zong & Porte-Agel (2020) Sec 4.1: nu_E = 0.03*Gamma0/(2*pi) = 0.005*Gamma0, taking
    the peak vortex-induced velocity Gamma0/(2*pi*Rv) and the core radius Rv as the
    reference scales. NuT_max*0.2 = 0.025*0.2 = 0.005 supplies that coefficient.
    """
    return NuT_max * 0.2 * jnp.abs(jnp.nansum(dgamma))

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

def _merge_close_vortices(vortex_field, threshold, dist_matrix):
    """Merge vortices that are closer than the threshold distance.
    """
    Circ0_full = vortex_field.Circ
    active_full = vortex_field.active
    N_cap = Circ0_full.shape[0] // 2

    Y0 = vortex_field.Y[:N_cap]
    Z0 = vortex_field.Z[:N_cap]
    Rv0 = vortex_field.Rv[:N_cap]
    Nu0 = vortex_field.Nu[:N_cap]
    Circ0 = Circ0_full[:N_cap]
    active0 = active_full[:N_cap]

    real_dist = dist_matrix[:N_cap, :N_cap]
    i_idx, j_idx = np.triu_indices(N_cap, k=1)  # static -- N_cap is a concrete Python int
    qualifies_all = real_dist[i_idx, j_idx] <= threshold**2

    def merge_step(carry, xs):
        Y, Z, Rv, Nu, claimed = carry
        i, j, qualifies = xs
        can_merge = qualifies & active0[i] & active0[j] & (~claimed[i]) & (~claimed[j])

        Circ_i, Circ_j = Circ0[i], Circ0[j]
        total_circ = Circ_i + Circ_j
        combine = jnp.abs(total_circ) > 1e-12
        denom = jnp.where(combine, total_circ, 1.0)
        merged_Y = (Y[i] * Circ_i + Y[j] * Circ_j) / denom
        merged_Z = (Z[i] * Circ_i + Z[j] * Circ_j) / denom

        w_i, w_j = jnp.abs(Circ_i), jnp.abs(Circ_j)
        w_sum = jnp.where(w_i + w_j > 1e-12, w_i + w_j, 1.0)
        merged_Rv = jnp.sqrt((w_i * Rv[i]**2 + w_j * Rv[j]**2) / w_sum)
        merged_Nu = (w_i * Nu[i] + w_j * Nu[j]) / w_sum  # same |Circ| weighting as Rv

        do_combine = can_merge & combine
        do_remove_both = can_merge & (~combine)

        Y = Y.at[j].set(jnp.where(do_combine, merged_Y, Y[j]))
        Z = Z.at[j].set(jnp.where(do_combine, merged_Z, Z[j]))
        Rv = Rv.at[j].set(jnp.where(do_combine, merged_Rv, Rv[j]))
        Nu = Nu.at[j].set(jnp.where(do_combine, merged_Nu, Nu[j]))
        claimed = claimed.at[i].set(claimed[i] | can_merge)
        claimed = claimed.at[j].set(claimed[j] | do_remove_both)
        return (Y, Z, Rv, Nu, claimed), None

    init_carry = (Y0, Z0, Rv0, Nu0, jnp.zeros(N_cap, dtype=bool))
    (Y, Z, Rv, Nu, claimed), _ = lax.scan(merge_step, init_carry, (i_idx, j_idx, qualifies_all))

    active_new = active0 & (~claimed)
    Circ_new = jnp.where(claimed, 0.0, Circ0)
    Y_new = jnp.where(claimed, _GRAVEYARD, Y)
    Z_new = jnp.where(claimed, _GRAVEYARD, Z)

    return VortexField(
        Y=jnp.concatenate([Y_new, Y_new]),
        Z=jnp.concatenate([Z_new, -Z_new]),
        Rv=jnp.concatenate([Rv, Rv]),
        Nu=jnp.concatenate([Nu, Nu]),
        Circ=jnp.concatenate([Circ_new, -Circ_new]),
        active=jnp.concatenate([active_new, active_new]),
        yloc=vortex_field.yloc,
        zloc=vortex_field.zloc,
        V=vortex_field.V,
        W=vortex_field.W,
        OmegaX=vortex_field.OmegaX,
        t=vortex_field.t
    )

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

    In-place slice assignment (Z2[N_real:] = ...) isn't allowed on immutable jnp arrays --
    replaced with .at[].set(), which is the direct jax equivalent.
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
