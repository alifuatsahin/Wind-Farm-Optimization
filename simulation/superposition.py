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

def get_local_velocity_field(config, wind_farm, method='MCS'):
    """Compute the combined velocity field at a given downstream turbine plane."""

    upstream_turbines = [t for t in wind_farm.turbines if t.pos[0] < config.pos[0]]

    # Freestream inflow at that turbine plane
    U_base = config._init_Uin() # shape (Ny, Nz)
    V_base = np.zeros_like(U_base)
    W_base = np.zeros_like(U_base)

    if not upstream_turbines:
        return U_base, V_base, W_base

    # Get wake fields at this x-plane
    wake_fields = [interpolate_vortex_field(t, config.pos[0], config.yloc, config.zloc, config.pos, config._init_Uin()) 
                   for t in upstream_turbines]

    u_yz = np.array([wf.U for wf in wake_fields])
    v_yz = np.array([wf.V for wf in wake_fields])
    w_yz = np.array([wf.W for wf in wake_fields])

    # Superpose wakes
    U, V, W = superpose(U_base, u_yz, v_yz, w_yz, method=method)

    return U, V, W

def superpose(U_in, u_yz, v_yz=None, w_yz=None, method='MCS'):
    """Superpose multiple wake velocity fields using specified method."""
    if method == 'MCS':
        return momentum_conserving_superposition(U_in, u_yz, v_yz, w_yz)
    else:
        return RSS_superposition(U_in, u_yz, v_yz, w_yz)

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

def momentum_conserving_superposition(U_in, u_yz, v_yz=None, w_yz=None, max_iter=50, tol=1e-3):
    """
    Momentum-Conserving Superposition (MCS) of multiple wake velocity fields.
        U_in: Freestream velocity field (2D array of shape (Ny, Nz))
        u_yz: Wake velocity fields (3D array of shape (i_turbine, Ny, Nz))
        v_yz: Transverse wake velocity fields (3D array of shape (i_turbine, Ny, Nz)) or None
        w_yz: Vertical wake velocity fields (3D array of shape (i_turbine, Ny, Nz)) or None
        Returns combined wake velocity field (2D array of shape (Ny, Nz))
    """

    # 1. Calculate Individual Deficits (u_i_s)
    u_s = np.maximum(U_in[None, :, :] - u_yz, 0) # shape (i_turbine, Ny, Nz)

    # 2. Calculate Individual Convection Velocities (Uc_i)
    u_c = np.sum(u_yz * u_s, axis=(1,2)) / np.sum(u_s, axis=(1,2)) # shape (i_turbine,)

    U_c = np.max(u_c, axis=0)  # initial guess for Uc

    # ---- ITERATION LOOP (Eq 2.7 & 2.9) ----
    for _ in range(max_iter):
        U_c = np.maximum(U_c, 1e-6)

        weights = u_c / U_c # shape (i_turbine,)
        # if sum(weights) > len(weights) + 1e-3:
        # weights_sum = sum(weights)
        # weights = [w / weights_sum for w in weights]  # normalize weights
        U_s = np.sum(weights[:, None, None] * u_s, axis=0)  # shape (Ny, Nz)
        U_s = np.minimum(U_s, U_in)  # prevent over-deficit

        U = U_in - U_s
        Uc_new = np.sum(U * U_s, axis=(0,1)) / np.sum(U_s, axis=(0,1))

        if np.abs(Uc_new - U_c) / np.abs(Uc_new) < tol:
            U_c = Uc_new
            break
        U_c = Uc_new

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
    