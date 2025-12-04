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
        return interp_field(vortex_data_list[0].U, source_yloc, source_zloc, yloc, zloc, default=default)
    if X >= positions[-1]:
        return interp_field(vortex_data_list[-1].U, source_yloc, source_zloc, yloc, zloc, default=default)
    
    idx = np.searchsorted(positions, X)
    i0 = idx - 1
    i1 = idx
    X0, X1 = positions[i0], positions[i1]
    # safe alpha (handles zero interval)
    alpha = 0.0 if X1 == X0 else (X - X0) / (X1 - X0)

    d0, d1 = vortex_data_list[i0], vortex_data_list[i1]

    # linear interp of arrays (works when shapes match)
    U = (1 - alpha) * d0.U + alpha * d1.U

    Uinterp = interp_field(U, source_yloc, source_zloc, yloc, zloc, default=default)
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

    Uinterp = interp_field(U, source_yloc, source_zloc, target_yloc, target_zloc, default=default)
    Vinterp = interp_field(V, source_yloc, source_zloc, target_yloc, target_zloc)
    Winterp = interp_field(W, source_yloc, source_zloc, target_yloc, target_zloc)
    
    return VortexField(
        yloc=target_yloc,
        zloc=target_zloc,
        V=Vinterp,
        W=Winterp,
        U=Uinterp,
        X=X
    )

def interp_field(field, y_source, z_source, target_yloc, target_zloc, default=None):
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

def get_local_velocity_field(config, wind_farm):
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

    # Momentum-conserving superposition
    U_wake, V_wake, W_wake, _ = momentum_conserving_superposition(
        U_in=U_base, U_list=U_list, V_list=V_list, W_list=W_list, plot=True
    )

    # Total local wind = freestream + wake-induced perturbation
    U_total = U_wake
    V_total = V_wake
    W_total = W_wake

    return U_total, V_total, W_total

def momentum_conserving_superposition(U_in, U_list, V_list=None, W_list=None, max_iter=50, tol=1e-3, plot=False):
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
    Uc_history = [Uc]
    Us_history = []

    # ---- ITERATION LOOP (Eq 2.7 & 2.9) ----
    for i in range(max_iter):
        if Uc < 1e-6: Uc = 1e-6

        weights = [Uc_i / Uc for Uc_i in Uc_list]
        Us_total = sum(w * u_s for w, u_s in zip(weights, u_s_list))
        Us_total = np.minimum(Us_total, U_in) 
        Us_total = np.maximum(Us_total, 0)
        Us_history.append(Us_total)

        Uw_total = U_in - Us_total
        
        num = np.sum(Uw_total * Us_total)
        den = np.sum(Us_total)

        if den < 1e-8:
            break

        Uc_new = num / den
        Uc_history.append(Uc_new)

        if np.abs(Uc_new - Uc) / np.abs(Uc_new) < tol:
            Uc = Uc_new
            break
        Uc = Uc_new

    # Final calculation
    weights = [Uc_i / Uc for Uc_i in Uc_list]
    Us_total = sum(w * u_s for w, u_s in zip(weights, u_s_list))
    Us_total = np.minimum(Us_total, U_in) 
    Us_total = np.maximum(Us_total, 0)
    U_total = U_in - Us_total

    if plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1.plot(Uc_history, marker='o')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Convection Velocity Uc')
        ax1.set_title('Convection Velocity Convergence')
        ax1.grid()
        ax2.plot(Us_history[-1][:, Us_history[-1].shape[1]//2])
        ax2.set_xlabel('Y Index')
        ax2.set_ylabel('Final Total Deficit Us')
        ax2.set_title('Final Total Deficit Profile at Centerline')
        ax2.grid()
        plt.tight_layout()
        plt.show()

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

    return U_total, V_total, W_total, Uc
    