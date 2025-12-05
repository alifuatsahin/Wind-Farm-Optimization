import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .data_structures import VortexField

def interpolate_vec_data(vortex_data_list, t):
    """
    Interpolate vortex field at time t from a list of VortexField objects.
    vortex_data_list must be time-sorted.
    Caveat: works best when vortex arrays correspond between frames.
    """
    times = np.array([v.t for v in vortex_data_list])

    if t <= times[0]:
        return vortex_data_list[0]
    if t >= times[-1]:
        return vortex_data_list[-1]
    
    idx = np.searchsorted(times, t)
    i0 = idx - 1
    i1 = idx
    t0, t1 = times[i0], times[i1]
    # safe alpha (handles zero interval)
    alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)

    d0, d1 = vortex_data_list[i0], vortex_data_list[i1]

    # linear interp of arrays (works when shapes match)
    V = (1 - alpha) * d0.V + alpha * d1.V
    W = (1 - alpha) * d0.W + alpha * d1.W

    return VortexField(
        Y=d0.Y,
        Z=d0.Z,
        Rv=d0.Rv,
        Circ=d0.Circ,
        yloc=d0.yloc,
        zloc=d0.zloc,
        V=V,
        W=W,
        OmegaX=d0.OmegaX,
        t=t
    )

def interpolate_local_velocity_field(turbine, X, yloc, zloc, default):
    """Interpolate vortex field at position X from a list of VortexField objects."""
    vortex_data_list = turbine.wake_field

    positions = np.array([v.X for v in vortex_data_list])

    source_yloc = turbine.yloc + turbine.pos[1]
    source_zloc = turbine.zloc + turbine.pos[2]

    if X <= positions[0]:
        return _interp_field(vortex_data_list[0].U, source_yloc, source_zloc, yloc, zloc, default=default)
    if X >= positions[-1]:
        return _interp_field(vortex_data_list[-1].U, source_yloc, source_zloc, yloc, zloc, default=default)
    
    idx = np.searchsorted(positions, X)
    i0 = idx - 1
    i1 = idx
    X0, X1 = positions[i0], positions[i1]
    # safe alpha (handles zero interval)
    alpha = 0.0 if X1 == X0 else (X - X0) / (X1 - X0)

    d0, d1 = vortex_data_list[i0], vortex_data_list[i1]

    # linear interp of arrays (works when shapes match)
    U = (1 - alpha) * d0.U + alpha * d1.U

    Uinterp = _interp_field(U, source_yloc, source_zloc, yloc, zloc, default=default)
    return Uinterp

def interpolate_vortex_field(turbine, X, yloc, zloc, target_pos, default):
    """Interpolate vortex field at position X from a list of VortexField objects."""
    vortex_data_list = turbine.wake_field

    positions = np.array([v.X for v in vortex_data_list])

    if X <= positions[0]:
        return vortex_data_list[0]
    if X >= positions[-1]:
        return vortex_data_list[-1]
    
    idx = np.searchsorted(positions, X)
    i0 = idx - 1
    i1 = idx
    X0, X1 = positions[i0], positions[i1]
    # safe alpha (handles zero interval)
    alpha = 0.0 if X1 == X0 else (X - X0) / (X1 - X0)

    d0, d1 = vortex_data_list[i0], vortex_data_list[i1]

    # linear interp of arrays (works when shapes match)
    V = (1 - alpha) * d0.V + alpha * d1.V
    W = (1 - alpha) * d0.W + alpha * d1.W
    U = (1 - alpha) * d0.U + alpha * d1.U

    source_yloc = turbine.yloc + turbine.pos[1]
    source_zloc = turbine.zloc + turbine.pos[2]

    target_yloc = yloc + target_pos[1]
    target_zloc = zloc + target_pos[2]

    Uinterp = _interp_field(U, source_yloc, source_zloc, target_yloc, target_zloc, default=default)
    Vinterp = _interp_field(V, source_yloc, source_zloc, target_yloc, target_zloc)
    Winterp = _interp_field(W, source_yloc, source_zloc, target_yloc, target_zloc)
    
    return VortexField(
        yloc=target_yloc,
        zloc=target_zloc,
        V=Vinterp,
        W=Winterp,
        U=Uinterp,
        X=X
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
        # If no default, replace NaNs with 0.0 or keep them
        result = np.nan_to_num(result, nan=0.0)
    return result

def get_local_velocity_field(config, wind_farm, superposition_method='MCS'):
    """Compute the combined velocity field at a given downstream turbine plane."""

    upstream_turbines = [t for t in wind_farm.turbines if t.pos[0] < config.pos[0]]

    # Freestream inflow at that turbine plane
    U_base = config._init_Uin()
    V_base = np.zeros_like(U_base)
    W_base = np.zeros_like(U_base)

    if not upstream_turbines:
        return U_base, V_base, W_base

    # Get wake fields at this x-plane
    wake_fields = [interpolate_vortex_field(t, config.pos[0], config.yloc, config.zloc, config.pos, config._init_Uin()) 
                   for t in upstream_turbines]

    U_list = [wf.U for wf in wake_fields]
    V_list = [wf.V for wf in wake_fields]
    W_list = [wf.W for wf in wake_fields]

    if superposition_method == 'MCS':
        # Momentum-conserving superposition
        U_wake, V_wake, W_wake = momentum_conserving_superposition(
            U_in=U_base, U_list=U_list, V_list=V_list, W_list=W_list
        )
    else:
        # Root-Sum-Square superposition
        U_wake, V_wake, W_wake = RSS_superposition(
            U_in=U_base, U_list=U_list, V_list=V_list, W_list=W_list
        )

    # Total local wind = freestream + wake-induced perturbation
    U_total = U_wake
    V_total = V_wake
    W_total = W_wake

    return U_total, V_total, W_total

def RSS_superposition(U_in, U_list, V_list=None, W_list=None):
    """
    Root-Sum-Square (RSS) superposition of multiple wake velocity fields.
    U_in: Freestream velocity field (2D array)
    U_list: List of wake velocity fields (2D arrays)
    Returns combined wake velocity field (2D array)
    """
    # Convert to arrays
    U_list = [np.asarray(U) for U in U_list]
    
    # 1. Calculate Individual Deficits (u_i_s)
    u_s_list = [np.maximum(U_in - U, 0) for U in U_list]

    # 2. Calculate Total Deficit (Eq 2.4)
    Us_total = np.sqrt(np.sum([np.minimum(u_s, 100) ** 2 for u_s in u_s_list], axis=0))
    Us_total = np.minimum(Us_total, U_in * 0.99)  # prevent over-deficit

    # 3. Combined Wake Velocity Field
    Uw_total = U_in - Us_total

    if V_list is not None:
        V_list = [np.asarray(V) for V in V_list]
        V_total = np.sqrt(np.sum([V**2 for V in V_list], axis=0))
    else:
        V_total = None
    if W_list is not None:
        W_list = [np.asarray(W) for W in W_list]
        W_total = np.sqrt(np.sum([W**2 for W in W_list], axis=0))
    else:
        W_total = None

    return Uw_total, V_total, W_total

def momentum_conserving_superposition(U_in, U_list, V_list=None, W_list=None, max_iter=50, tol=1e-3):
    # Convert to arrays
    U_list = [np.asarray(U) for U in U_list]
    
    # 1. Calculate Individual Deficits (u_i_s)
    # Ensure no negative deficits due to numerical noise if U > U_in slightly
    u_s_list = [np.maximum(U_in - U, 0) for U in U_list]

    # 2. Calculate Individual Convection Velocities (Uc_i)
    Uc_list = []
    for U, u_s in zip(U_list, u_s_list):
        den = np.sum(u_s)
        if den < 1e-8:
            Uc_list.append(np.mean(U_in))
        else:
            num = np.sum(U * u_s)
            Uc_val = num / den
            Uc_list.append(Uc_val)

    Uc = np.max(Uc_list) if Uc_list else np.mean(U_in)

    # ---- ITERATION LOOP (Eq 2.7 & 2.9) ----
    for i in range(max_iter):
        if Uc < 1e-6: Uc = 1e-6

        weights = [Uc_i / Uc for Uc_i in Uc_list]
        # if sum(weights) > len(weights) + 1e-3:
        # weights_sum = sum(weights)
        # weights = [w / weights_sum for w in weights]  # normalize weights
        Us_total = sum(w * u_s for w, u_s in zip(weights, u_s_list))
        Us_total = np.minimum(Us_total, U_in * 0.99)  # prevent over-deficit

        Uw_total = U_in - Us_total
        
        num = np.sum(Uw_total * Us_total)
        den = np.sum(Us_total)

        if den < 1e-8:
            break

        Uc_new = num / den
        relaxation = 0.05
        Uc_new = (1 - relaxation) * Uc + (relaxation * Uc_new)

        if np.abs(Uc_new - Uc) / np.abs(Uc_new) < tol:
            Uc = Uc_new
            break
        Uc = Uc_new

    # Transverse
    if V_list is not None:
        V_list = [np.asarray(V) for V in V_list]
        V_total = sum((Uc_i / Uc) * V for Uc_i, V in zip(Uc_list, V_list))
    else:
        V_total = None
        
    if W_list is not None:
        W_list = [np.asarray(W) for W in W_list]
        W_total = sum((Uc_i / Uc) * W for Uc_i, W in zip(Uc_list, W_list))
    else:
        W_total = None

    return Uw_total, V_total, W_total
    