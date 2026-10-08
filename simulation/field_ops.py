"""Interpolation of a turbine's marched wake field onto arbitrary grids.
"""
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator

from .data_structures import VortexField

def interpolate_vec_data(stacked, t):
    """
    Interpolate vortex field at time t from `stacked`, a VortexField whose every array leaf
    has a leading (total_steps,) frame-index axis (Loop 1's raw, UNTRIMMED jit output -- see
    vortex_model._simulate_vortex_evolution_jit) instead of a Python list of VortexField
    objects. `stacked.t` must be non-decreasing along that axis.
    """
    t_stack = stacked.t
    n = t_stack.shape[0]

    below = t <= t_stack[0]
    above = t >= t_stack[-1]

    idx = jnp.searchsorted(t_stack, t)
    idx_safe = jnp.clip(idx, 1, n - 1)
    i0 = idx_safe - 1
    i1 = idx_safe
    t0, t1 = t_stack[i0], t_stack[i1]
    denom = t1 - t0
    same_t = denom == 0
    alpha = jnp.where(same_t, 0.0, (t - t0) / jnp.where(same_t, 1.0, denom))
    source_idx = jnp.where(below, 0, jnp.where(above, n - 1, i0))

    def gather(leaf):
        return leaf[source_idx]

    def interp_pair(leaf):
        blended = (1 - alpha) * leaf[i0] + alpha * leaf[i1]
        return jnp.where(below | above, leaf[source_idx], blended)

    return VortexField(
        Y=gather(stacked.Y), Z=gather(stacked.Z), Rv=gather(stacked.Rv), Circ=gather(stacked.Circ),
        age=gather(stacked.age), active=gather(stacked.active),
        # yloc/zloc are the cross-plane grid and carry no frame axis -- they are identical
        # at every step, so the vortex march stores one copy rather than total_steps of
        # them. Passing them through is exactly what gathering used to return.
        yloc=stacked.yloc, zloc=stacked.zloc,
        V=interp_pair(stacked.V), W=interp_pair(stacked.W),
        OmegaX=gather(stacked.OmegaX),
        t=t,
    )

def interpolate_local_velocity_field(turbine, X, yloc, zloc, default):
    """
    Interpolate wake velocity field at streamwise position X onto target grid.
    
    Args:
        turbine: Turbine object containing wake_field data
        X: Streamwise position (global coordinates) where to interpolate
        yloc: Target lateral grid coordinates (2D array)
        zloc: Target vertical grid coordinates (2D array)
        default: Default velocity field for points outside interpolation bounds
    
    Returns:
        Uinterp: Interpolated streamwise velocity field on target grid (yloc, zloc)
        Uin_interp: Interpolated local input velocity field on target grid (yloc, zloc)
    """
    vortex_data_list = turbine.wake_field

    positions = np.array([v.X for v in vortex_data_list])

    source_yloc = turbine.yloc + turbine.pos[1]
    source_zloc = turbine.zloc + turbine.pos[2]

    if X <= positions[0]:
        boundary = vortex_data_list[0]
    elif X >= positions[-1]:
        boundary = vortex_data_list[-1]
    else:
        boundary = None

    if boundary is not None:
        Uinterp = _interp_field(boundary.U, source_yloc, source_zloc, yloc, zloc, default=default)
        Uin_interp = _interp_field(turbine.Uin, source_yloc, source_zloc, yloc, zloc, default=default)
        return Uinterp, Uin_interp

    idx = np.searchsorted(positions, X)
    i0 = idx - 1
    i1 = idx
    X0, X1 = positions[i0], positions[i1]
    alpha = 0.0 if X1 == X0 else (X - X0) / (X1 - X0)

    d0, d1 = vortex_data_list[i0], vortex_data_list[i1]

    # linear interp of arrays
    U = (1 - alpha) * d0.U + alpha * d1.U

    Uinterp = _interp_field(U, source_yloc, source_zloc, yloc, zloc, default=default)
    Uin_interp = _interp_field(turbine.Uin, source_yloc, source_zloc, yloc, zloc, default=default)

    return Uinterp, Uin_interp

def interpolate_vortex_field(turbine, target_pos, yloc, zloc, default):
    """Interpolate vortex field at position X from a list of VortexField objects."""
    vortex_data_list = turbine.wake_field
    X = target_pos[0] - turbine.pos[0]

    positions = np.array([v.X for v in vortex_data_list])

    source_yloc = turbine.yloc + turbine.pos[1]
    source_zloc = turbine.zloc + turbine.pos[2]

    target_yloc = yloc + target_pos[1]
    target_zloc = zloc + target_pos[2]

    # Determine which wake field to use
    if X <= positions[0]:
        d = vortex_data_list[0]
        U, V, W = d.U, d.V, d.W
    elif X >= positions[-1]:
        d = vortex_data_list[-1]
        U, V, W = d.U, d.V, d.W
    else:
        # Interpolate in X direction
        idx = np.searchsorted(positions, X)
        i0 = idx - 1
        i1 = idx
        X0, X1 = positions[i0], positions[i1]
        alpha = 0.0 if X1 == X0 else (X - X0) / (X1 - X0)

        d0, d1 = vortex_data_list[i0], vortex_data_list[i1]
        V = (1 - alpha) * d0.V + alpha * d1.V
        W = (1 - alpha) * d0.W + alpha * d1.W
        U = (1 - alpha) * d0.U + alpha * d1.U

    Uinterp = _interp_field(U, source_yloc, source_zloc, target_yloc, target_zloc, default=default)
    Vinterp = _interp_field(V, source_yloc, source_zloc, target_yloc, target_zloc)
    Winterp = _interp_field(W, source_yloc, source_zloc, target_yloc, target_zloc)
    
    return VortexField(
        yloc=target_yloc,
        zloc=target_zloc,
        V=Vinterp,
        W=Winterp,
        U=Uinterp,
        X=target_pos[0]
    )

def _interp_field(field, y_source, z_source, target_yloc, target_zloc, default=None):
    interp = RegularGridInterpolator(
        (y_source[:,0], z_source[0,:]),
        field,
        bounds_error=False,
        fill_value=np.nan
    )
    points = np.vstack([target_yloc.ravel(), target_zloc.ravel()]).T
    result = interp(points).reshape(target_yloc.shape)
    if default is not None:
        # fill points outside original grid with freestream (default)
        mask = np.isnan(result)
        result[mask] = default[mask]
    else:
        # If no default, replace NaNs with 0.0
        result = np.nan_to_num(result, nan=0.0)
    return result


# ------------------------------------------------------------------ field read-out
# In a single marched field the velocity at any station is whatever the nearest upstream
# rotor's march carried there -- no combination step, so these replace what used to be
# superpose() over every turbine's wake.

def _source_turbine(turbines, x_m):
    return max((t for t in turbines if t.pos[0] < x_m), key=lambda t: t.pos[0], default=None)


def _log_profile(fp, Z):
    return fp.Uh * (np.log(np.maximum(Z, fp.z0 + 1e-3) / fp.z0) / np.log(fp.Zh / fp.z0))


def extract_cross_plane(wind_farm, x_m, y_grid, z_grid):
    """Streamwise velocity on a (Y, Z) plane at one station. 1-D y/z vectors in."""
    fp = wind_farm.field_params
    Y, Z = np.meshgrid(y_grid, z_grid, indexing='ij')
    U_in = _log_profile(fp, Z)
    src = _source_turbine(wind_farm.turbines, x_m)
    if src is None:
        return U_in
    U, _ = interpolate_local_velocity_field(src, x_m - src.pos[0], Y, Z, default=U_in)
    return np.asarray(U)


def extract_hub_height_slice(wind_farm, x_m_target, y_m_target, nz=41):
    """Hub-height (x, y) map of streamwise velocity, plus the reference Uh."""
    ts = wind_farm.turbines
    fp = wind_farm.field_params
    D, Zhub = ts[0].D, ts[0].Zhub
    x_global, y_global = np.asarray(x_m_target), np.asarray(y_m_target)

    z_vec = np.linspace(0.0, max(t.Zhub + t.pos[2] for t in ts) + fp.max_Z * D, nz)
    Y, Z = np.meshgrid(y_global, z_vec, indexing='ij')
    k_hub = int(np.argmin(np.abs(z_vec - Zhub)))
    U_in = _log_profile(fp, Z)

    U_map = np.zeros((len(y_global), len(x_global)))
    for i, xg in enumerate(x_global):
        src = _source_turbine(ts, xg)
        if src is None:
            U_map[:, i] = U_in[:, k_hub]
            continue
        U, _ = interpolate_local_velocity_field(src, xg - src.pos[0], Y, Z, default=U_in)
        U_map[:, i] = np.asarray(U)[:, k_hub]
    return {'U': U_map, 'Uh': float(fp.Uh)}
