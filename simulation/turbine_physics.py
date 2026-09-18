import dataclasses
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .turbine_state import LocalConditions, VortexSimConfig, DeficitFieldConfig, pack_upstream_turbines
from .utils import smooth_2d, NuT_model
from .vortex_model import _simulate_vortex_evolution_jit
from .model_solver import advance_wake_field
from .superposition import interpolate_vec_data


def init_Uin(params):
    zsafe = np.maximum(params.zloc, params.field_params.z0 + 1e-6)  # avoid log(0) issues
    Uin = params.Uh * (np.log(zsafe / params.field_params.z0) / np.log(params.Zh / params.field_params.z0))
    return Uin


def nominal_hub_velocity(params):
    """Rotor-averaged undisturbed inflow -- depends only on config, never on the mutated Uin."""
    Uin = init_Uin(params)
    rotor_mask = np.sqrt((params.yloc) ** 2 + (params.zloc - params.Zhub) ** 2) <= (params.D / 2)
    return np.mean(Uin[rotor_mask])


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
    omega = compute_omega(params, Uhub)
    return np.pi * Uhub ** 2 * params.Ct / omega


def compute_Ut(params, Uhub):
    omega = compute_omega(params, Uhub)
    z = params.Zhub + params.D / 2.0 * np.sin(params.phi)
    zsafe = np.maximum(z, params.field_params.z0 + 1e-6)
    Utbl = params.Uh * (np.log(zsafe / params.field_params.z0) / np.log(params.Zh / params.field_params.z0))
    U_tx = (1 - params.a) * Utbl - omega * params.R * np.sin(params.phi) * np.sin(params.beta)
    U_ty = -omega * params.R * np.sin(params.phi) * np.cos(params.beta)
    U_tz = omega * params.R * np.cos(params.phi)
    return np.array([U_tx, U_ty, U_tz]).T


def compute_dgamma(params, Uhub):
    Ut = compute_Ut(params, Uhub)
    gamma0 = compute_gamma0(params, Uhub)
    alpha = np.arcsin(Ut[:, 0] / np.sqrt(np.sum(Ut ** 2, axis=1)))
    dgamma = np.sin(alpha) * params.dphi
    gamma_ref = gamma0 * 0.45
    dgamma = gamma_ref / np.sum(dgamma[1:]) * dgamma
    return dgamma


def calculate_efficiency(params, Uhub):
    P = (Uhub ** 3) * params.Cp
    nominal_P = (nominal_hub_velocity(params) ** 3) * params.config.Cp
    return P / nominal_P


def simulate_vortex_field(params, local):
    """Builds a VortexSimConfig"""
    adapter = VortexSimConfig(
        D=params.D,
        Uhub=local.Uhub,
        dgamma=jnp.asarray(compute_dgamma(params, local.Uhub)),
        calculation_domain=params.calculation_domain,
        phi=jnp.asarray(params.phi),
        beta=params.beta,
        Yoffset=params.Yoffset,
        Zhub=params.Zhub,
        Nv=params.Nv,
        V=jnp.asarray(local.V),
        W=jnp.asarray(local.W),
        yloc=jnp.asarray(params.yloc),
        zloc=jnp.asarray(params.zloc),
    )
    stacked, _was_active = _simulate_vortex_evolution_jit(
        adapter, params.field_params.NuT_max, params.field_params.merge_threshold,
        params.field_params.cfl_factor, 1000,
    )
    return stacked


def initialize_wake_field(params, stacked, local):
    yloc = np.asarray(stacked.yloc[0])
    zloc = np.asarray(stacked.zloc[0])
    beta = params.beta

    dl = params.dl  # equivalent to yloc[1,0]-yloc[0,0]; verified in Step 1 sub-step 5
    U = np.asarray(local.Uin).copy()
    mask = np.sqrt(((yloc + params.Yoffset) ** 2) / (np.cos(beta) ** 2) + (zloc - params.Zhub) ** 2) <= params.R
    U[mask] -= 2.0 * U[mask] * params.a

    hub_mask = (np.abs(yloc) <= dl * 1.0) & (zloc < params.Zhub)
    U[hub_mask] -= 0.3 * U[hub_mask]  # add some velocity deficit at the hub

    U_smooth = smooth_2d(U, kernel_size=3)

    total_steps = stacked.t.shape[0]
    U_field = jnp.zeros((total_steps,) + U_smooth.shape).at[0].set(jnp.asarray(U_smooth))

    new_stacked = dataclasses.replace(
        stacked,
        U=U_field,
        X=stacked.X.at[0].set(0.0),
        t=stacked.t.at[0].set(0.0),
    )
    return new_stacked, dl


def calculate_deficit_field(params, local, stacked, dl, upstream_turbines, N_upstream_max=None, max_steps=1500):
    if N_upstream_max is None:
        N_upstream_max = len(upstream_turbines)
    up_a, up_D, up_pos_x, up_Uhub, up_mask = pack_upstream_turbines(upstream_turbines, N_upstream_max)

    adapter = DeficitFieldConfig(
        pos=jnp.asarray(params.pos), D=params.D, Uhub=local.Uhub, Uin=jnp.asarray(local.Uin), Zhub=params.Zhub,
    )
    dt_cap = min(dl / local.Uhub, 0.25 * params.D / local.Uhub)  # constant for the whole run -- plain floats

    seed = jax.tree_util.tree_map(lambda leaf: leaf[0], stacked)
    stacked_out, was_active = _calculate_deficit_field_jit(
        stacked, seed, adapter, params.field_params.I_amb, params.field_params.WV,
        up_a, up_D, up_pos_x, up_Uhub, up_mask, dl, dt_cap, params.calculation_domain, max_steps,
    )
    # was_active[k] corresponds to buffer index k+1 (the loop below is 1-indexed exactly like
    # the original); kept_count includes the seed (index 0) plus every real advance.
    kept_count = int(jnp.sum(was_active)) + 1

    frames = [_vortex_field_to_numpy(seed)]
    frames += [
        _vortex_field_to_numpy(jax.tree_util.tree_map(lambda leaf, i=i: leaf[i], stacked_out))
        for i in range(kept_count - 1)
    ]
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
            new = dataclasses.replace(new, U=U, X=X_new, t=current.t + dt)

            still_active = new.X <= calculation_domain  # this crossing entry stays the final real one
            return new, still_active

        def when_inactive(_):
            return current, active

        next_state, next_active = lax.cond(active, when_active, when_inactive, operand=None)
        return (next_state, next_active), (next_state, active)

    init_carry = (seed, True)
    _, (stacked_out, was_active) = lax.scan(scan_step, init_carry, xs=None, length=max_steps)
    return stacked_out, was_active


def _vortex_field_to_numpy(vortex_field):
    """Converts every array field of a VortexField from jnp back to plain numpy, and t/X
    (0-d jnp arrays once produced inside jax-based code) back to Python floats -- the Loop-2/
    downstream (save_results, plotting) boundary contract."""
    return dataclasses.replace(
        vortex_field,
        Y=np.asarray(vortex_field.Y), Z=np.asarray(vortex_field.Z),
        Rv=np.asarray(vortex_field.Rv), Circ=np.asarray(vortex_field.Circ),
        active=np.asarray(vortex_field.active),
        yloc=np.asarray(vortex_field.yloc), zloc=np.asarray(vortex_field.zloc),
        V=np.asarray(vortex_field.V), W=np.asarray(vortex_field.W),
        U=np.asarray(vortex_field.U), OmegaX=np.asarray(vortex_field.OmegaX),
        t=float(vortex_field.t), X=float(vortex_field.X),
    )
