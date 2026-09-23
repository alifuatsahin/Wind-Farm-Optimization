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
        Nu=gather(stacked.Nu), active=gather(stacked.active),
        yloc=gather(stacked.yloc), zloc=gather(stacked.zloc),
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

def get_local_velocity_field(config, wind_farm, method='linear'):
    """Compute the combined velocity field at a given downstream turbine plane."""

    upstream_turbines = [t for t in wind_farm.turbines if t.pos[0] < config.pos[0]]

    # Freestream inflow at that turbine plane
    U_base = config.init_Uin() # shape (Ny, Nz)
    V_base = np.zeros_like(U_base)
    W_base = np.zeros_like(U_base)

    if not upstream_turbines:
        return U_base, V_base, W_base

    # Get wake fields at this x-plane (interpolated to downstream grid)
    wake_fields = [interpolate_vortex_field(t, config.pos, config.yloc, config.zloc, config.init_Uin()) 
                   for t in upstream_turbines]
    
    # Interpolate local_Uins to downstream grid to match u_yz spatial locations
    local_Uins_interpolated = []
    for t in upstream_turbines:
        source_yloc = t.yloc + t.pos[1]
        source_zloc = t.zloc + t.pos[2]
        target_yloc = config.yloc + config.pos[1]
        target_zloc = config.zloc + config.pos[2]
        Uin_interp = _interp_field(t.Uin, source_yloc, source_zloc, target_yloc, target_zloc, default=U_base)
        local_Uins_interpolated.append(Uin_interp)
    
    local_Uins = np.array(local_Uins_interpolated)
    u_yz = np.array([wf.U for wf in wake_fields])
    v_yz = np.array([wf.V for wf in wake_fields])
    w_yz = np.array([wf.W for wf in wake_fields])

    # Superpose wakes
    U, V, W = superpose(U_base, local_Uins, u_yz, v_yz, w_yz, method=method)

    return U, V, W

def superpose(U_in, local_Uins, u_yz, v_yz=None, w_yz=None, method='linear'):
    """Superpose multiple wake velocity fields using specified method.

        'linear' linear sum of local-inflow deficits (default; Zong's Method C)
        'MCS'    momentum-conserving superposition
        'RSS'    root-sum-square against freestream (Method B; unusable in deep arrays,
                 R2 -8.4 on the 8-turbine case -- it sums many large quadrature terms
                 against the freestream and collapses the field)

    MCS has the stronger theoretical claim -- it is the only one derived from momentum
    conservation, and it is the only one that measurably achieves it (combined wake
    momentum / sum of parts = 1.000, where linear loses 5-10%). It is nonetheless NOT the
    default, on measurement: on the 8-turbine case linear gives field R2 0.953 vs 0.920,
    per-turbine power RMSE 0.085 vs 0.108, and LES R2 0.929 vs 0.923.

    The reason is a structural artifact of MCS at the first genuinely superposed station.
    There, two strong wakes overlap and nothing weak dilutes the deficit-weighted mean, so
    U_c (Eq 2.7) comes out low -- 0.708*U_in vs 0.761-0.777 further downstream -- the
    Eq 2.9 weights run high, and the combined deficit over-deepens. Turbine 3's predicted
    power dips to 0.86 of measured while its neighbours sit at 1.06 and 1.14. MCS's
    slightly better MEAN power bias (+6.3% vs +7.2%) is that dip cancelling against
    over-prediction elsewhere, not better accuracy -- judge on power RMSE, not the mean.
    """
    if method == 'linear':
        return linear_local_superposition(U_in, local_Uins, u_yz, v_yz, w_yz)
    elif method == 'MCS':
        return momentum_conserving_superposition(U_in, local_Uins, u_yz, v_yz, w_yz)
    elif method == 'RSS':
        return RSS_superposition(U_in, u_yz, v_yz, w_yz)
    else:
        raise ValueError(f"unknown superposition method {method!r}; expected 'linear', 'MCS' or 'RSS'")

def linear_local_superposition(U_in, local_Uins, u_yz, v_yz=None, w_yz=None):
    """
    Linear superposition of local-inflow deficits -- Zong & Porte-Agel's method C
    (Niayifar & Porte-Agel 2016).

    U_in: Freestream velocity field (2D array of shape (Ny, Nz))
    local_Uins: Freestream velocities at each turbine (3D array of shape (i_turbine, Ny, Nz))
    u_yz: Wake velocity fields (3D array of shape (i_turbine, Ny, Nz))
    Returns combined wake velocity field (2D array of shape (Ny, Nz))
    """

    # 1. Calculate Individual Deficits (u_i_s)
    u_s = np.maximum(local_Uins - u_yz, 0)

    # 2. Calculate Total Deficit
    U_s = np.sum(u_s, axis=0)
    U_s = np.minimum(U_s, U_in)  # prevent over-deficit

    # 3. Combined Wake Velocity Field
    U = U_in - U_s

    if v_yz is not None:
        V = np.sum(v_yz, axis=0)
    else:
        V = None
    if w_yz is not None:
        W = np.sum(w_yz, axis=0)
    else:
        W = None

    return U, V, W

def RSS_superposition(U_in, u_yz, v_yz=None, w_yz=None):
    """
    Root-Sum-Square (RSS) superposition of multiple wake velocity fields.
    U_in: Freestream velocity field (2D array of shape (Ny, Nz))
    u_yz: Wake velocity fields (3D array of shape (i_turbine, Ny, Nz))
    Returns combined wake velocity field (2D array of shape (Ny, Nz))
    """
    
    # 1. Calculate Individual Deficits (u_i_s)
    u_s = np.maximum(U_in[None, :, :] - u_yz, 0)

    # 2. Calculate Total Deficit (Eq 2.4)
    U_s = np.sqrt(np.sum(u_s ** 2, axis=0))
    U_s = np.minimum(U_s, U_in)  # prevent over-deficit

    # 3. Combined Wake Velocity Field
    U = U_in - U_s

    if v_yz is not None:
        V = np.sum(v_yz, axis=0)
    else:
        V = None
    if w_yz is not None:
        W = np.sum(w_yz, axis=0)
    else:
        W = None

    return U, V, W

def momentum_conserving_superposition(U_in, local_Uins, u_yz, v_yz=None, w_yz=None):
    """
    Momentum-Conserving Superposition (MCS) of multiple wake velocity fields.
        U_in: Freestream velocity field (2D array of shape (Ny, Nz))
        local_Uins: Freestream velocities at each turbine (3D array of shape (i_turbine, Ny, Nz))
        u_yz: Wake velocity fields (3D array of shape (i_turbine, Ny, Nz))
        v_yz: Transverse wake velocity fields (3D array of shape (i_turbine, Ny, Nz)) or None
        w_yz: Vertical wake velocity fields (3D array of shape (i_turbine, Ny, Nz)) or None
        Returns combined wake velocity field (2D array of shape (Ny, Nz))

    Eq (2.9) is solved in closed form. u_c^i and u_s^i are fixed, so with
    A = sum_i u_c^i u_s^i the update Uc <- sum(U*U_s)/sum(U_s) becomes
    Uc = P/R - (Q/R)/Uc for P = sum(U_in*A), Q = sum(A^2), R = sum(A), i.e. a root of

        Uc^2 - (P/R) Uc + Q/R = 0

    with the larger root the stable one. A negative discriminant means no
    momentum-conserving combined wake exists.

    Writing <f>_A = sum(f*A)/sum(A), the quadratic is Uc^2 - <U_in>_A Uc + <A>_A = 0, so a
    root exists iff <U_in>_A^2 >= 4<A>_A. For a SINGLE wake with u_c ~ U_in - u_s that
    reduces to (2r - 1)^2 >= 0 with r = u_s/U_in: always satisfied, touching zero exactly
    at r = 1/2. No-root is therefore a pure superposition effect, triggered once the
    COMBINED deficit passes roughly half the local inflow.

    Re-measured 2026-09-23 (after the d2Uin_dY2 solver fix, which invalidated the earlier
    ~4% figure): 7.1% of multi-wake stations on the 8-turbine case, 9.9% on the aligned
    LES, 0% on the yawed LES. In every case it fires ONLY where exactly two wakes overlap,
    and only in a ~2D window immediately behind the second rotor.

    This is a limit of MCS, not a defect in the wakes being fed to it. At those same
    stations the reference data carries a peak deficit of 0.61 (LES) and 0.75 (PIV) versus
    the model's summed 0.63 and 0.68 -- the flow really is that slow there, so the model is
    not over-removing momentum. Zong never hits it because his individual wakes are the
    smooth far-wake Gaussian of Eq 2.4, whose peak stays well under U_in/2; ours is a
    marched PVT field that resolves the near wake, where a deficit above U_in/2 is real.

    The discriminant is clamped to zero rather than branching to another superposition
    law. That puts U_c at the parabola vertex P/(2R), which is exactly where the root
    goes as the discriminant approaches zero, so the solution stays CONTINUOUS -- it
    matters because this feeds a layout optimizer. Branching instead (to linear, or worse
    to RSS) jumps the field by 19-24% of U_in at the switch. Measured cost of continuity:
    0.012 in field R2, and none at all in predicted power.
    """

    # 1. Calculate Individual Deficits (u_i_s)
    u_s = np.maximum(local_Uins - u_yz, 0) # shape (i_turbine, Ny, Nz)

    deficit_sums = np.sum(u_s, axis=(1,2))
    valid_mask = deficit_sums > 1e-6
    n_valid = int(np.count_nonzero(valid_mask))

    if n_valid == 0:
        # No valid wakes, return freestream
        V = np.sum(v_yz, axis=0) if v_yz is not None else None
        W = np.sum(w_yz, axis=0) if w_yz is not None else None
        return U_in, V, W

    if n_valid == 1:
        # A single wake superposed with nothing is itself
        weights = valid_mask.astype(float)
        U_s = np.minimum(np.sum(weights[:, None, None] * u_s, axis=0), U_in)
        V = np.sum(weights[:, None, None] * v_yz, axis=0) if v_yz is not None else None
        W = np.sum(weights[:, None, None] * w_yz, axis=0) if w_yz is not None else None
        return U_in - U_s, V, W

    # 2. Calculate Individual Convection Velocities (Uc_i)
    u_c = np.zeros(u_s.shape[0])
    u_c[valid_mask] = np.sum(u_yz * u_s, axis=(1,2))[valid_mask] / deficit_sums[valid_mask] # shape (i_turbine,)

    # 3. Solve Eq (2.9) for the combined convection velocity Uc
    A = np.tensordot(u_c, u_s, axes=(0, 0))  # shape (Ny, Nz)
    P = np.sum(U_in * A)
    Q = np.sum(A ** 2)
    R = np.sum(A)
    R = R if abs(R) > 1e-12 else 1e-12
    discriminant = (P / R) ** 2 - 4.0 * Q / R

    # Clamped, not branched -- see docstring. Continuous through discriminant = 0.
    U_c = 0.5 * (P / R + np.sqrt(max(discriminant, 0.0)))

    # 4. Combine using the momentum-conserving weights (Eq 2.7)
    weights = u_c / max(U_c, 1e-6) # shape (i_turbine,)
    U_s = np.sum(weights[:, None, None] * u_s, axis=0)  # shape (Ny, Nz)
    U_s = np.minimum(U_s, U_in)  # prevent over-deficit

    U = U_in - U_s

    # Transverse
    if v_yz is not None:
        V = np.sum(weights[:, None, None] * v_yz, axis=0)
    else:
        V = None

    if w_yz is not None:
        W = np.sum(weights[:, None, None] * w_yz, axis=0)
    else:
        W = None

    return U, V, W
