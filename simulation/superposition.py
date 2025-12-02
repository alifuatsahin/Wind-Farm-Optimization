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

# def momentum_conserving_superposition(U_in, U_list, V_list=None, W_list=None, max_iter=10, tol=1e-3):
#     """
#     Momentum-conserving superposition for U, V, W at a single x-plane.

#     Inputs:
#         U_list: list of 2D arrays (U_i of each turbine at this x-plane)
#         V_list: list of 2D arrays (V_i of each turbine at this x-plane)
#         W_list: list of 2D arrays (W_i of each turbine at this x-plane)
#         U_in : free-stream velocity
#     Returns:
#         U_total, V_total, W_total, Uc (combined convection velocity)
#     """

#     # Convert to arrays
#     U_list = [np.asarray(U) for U in U_list]

#     # Individual velocity deficits u_i_s = U_in - U
#     u_s_list = [U_in - U for U in U_list]

#     # # --- Initial convection velocity estimate: Uc_i = mean(U_i) ---
#     Uc_list = [np.mean(U) for U in U_list]
#     # Uc_list = []
#     # for U, u_s in zip(U_list, u_s_list):
#     #     # Denominator: Total Deficit Integral
#     #     den = np.sum(u_s)
        
#     #     if den < 1e-6:
#     #         # If there is no wake (deficit is 0), Uc is just freestream
#     #         Uc_list.append(np.mean(U_in))
#     #     else:
#     #         # Numerator: Momentum Flux (U_wake * Deficit)
#     #         num = np.sum(U * u_s)
#     #         Uc_val = num / den
#     #         Uc_list.append(Uc_val)

#     # Initial combined convection velocity: max(Uc_i)
#     Uc = np.max(Uc_list)

#     # ---- ITERATION LOOP (eq 2.7 & 2.9) ----
#     for _ in range(max_iter):

#         # eq 2.9: weighted deficit
#         weights = [Uc_i / Uc for Uc_i in Uc_list]
#         Us_total = sum(w * u_s for w, u_s in zip(weights, u_s_list))

#         # eq 2.7: update Uc = ∫ Uw * Us / ∫ Us
#         Uw_total = U_in - Us_total  # reminder: Us = U_inf - U
#         num = np.sum(Uw_total * Us_total)
#         den = np.sum(Us_total)

#         if den == 0:
#             break

#         Uc_new = num / den

#         if np.abs(Uc_new - Uc) / np.abs(Uc_new) < tol:
#             Uc = Uc_new
#             break

#         Uc = Uc_new

#     # Recalculate the final weighted deficit (Uw_total_final)
#     weights = [Uc_i / Uc for Uc_i in Uc_list]
#     Us_total = sum(w * u_s for w, u_s in zip(weights, u_s_list))
#     U_total = U_in - Us_total

#     # ---- Transverse velocities: V, W ----
#     # Use same weights as eq 4.1:   V = Σ (Uc_i / Uc) * V_i
#     if V_list is not None:
#         V_list = [np.asarray(V) for V in V_list]
#         V_total = sum((Uc_i / Uc) * V for Uc_i, V in zip(Uc_list, V_list))
#     else:
#         V_total = None
#     if W_list is not None:
#         W_list = [np.asarray(W) for W in W_list]
#         W_total = sum((Uc_i / Uc) * W for Uc_i, W in zip(Uc_list, W_list))
#     else:
#         W_total = None

#     return U_total, V_total, W_total, Uc

def momentum_conserving_superposition(U_in, U_list, V_list=None, W_list=None, max_iter=10, tol=1e-3, plot=False):
    """
    Momentum-conserving superposition implementing Eq 2.7 and 2.9.
    """

    # Convert to arrays
    U_list = [np.asarray(U) for U in U_list]
    
    # 1. Calculate Individual Deficits (u_i_s)
    u_s_list = [np.maximum(U_in - U, 0) for U in U_list]

    # 2. Calculate Individual Convection Velocities (Uc_i)
    Uc_list = []
    for U, u_s in zip(U_list, u_s_list):
        
        # Denominator: Integral of deficit (sum works because dA cancels out in ratio)
        den = np.sum(u_s)
        
        if den < 1e-8:
            # If integral is 0 (no wake at this slice), convection velocity is Freestream
            Uc_list.append(np.mean(U_in))
        else:
            # Numerator: Integral of (Velocity * Deficit)
            num = np.sum(U * u_s)
            Uc_val = num / den
            Uc_list.append(Uc_val)

    # Initial combined convection velocity: max(Uc_i) (As per text)
    Uc = np.max(Uc_list)

    # ---- ITERATION LOOP (Eq 2.7 & 2.9) ----
    for _ in range(max_iter):
        
        # Safety check to avoid division by zero if wakes are tiny
        if Uc < 1e-6: Uc = 1e-6

        # Eq 2.9: Calculate weighted total deficit
        # Us_total = Sum ( (Uc_i / Uc) * u_i_s )
        weights = [Uc_i / Uc for Uc_i in Uc_list]
        Us_total = sum(w * u_s for w, u_s in zip(weights, u_s_list))

        # Eq 2.7: Update Combined Convection Velocity (Uc)
        # Uc = Integral(Uw * Us) / Integral(Us)
        # Where Uw (Wake Velocity) = U_in - Us_total
        Uw_total = U_in - Us_total
        
        num = np.sum(Uw_total * Us_total)
        den = np.sum(Us_total)

        if den < 1e-8:
            break

        Uc_new = num / den

        # Check convergence
        if np.abs(Uc_new - Uc) / np.abs(Uc_new) < tol:
            Uc = Uc_new
            break

        Uc = Uc_new

    # Recalculate the final weighted deficit with converged Uc
    weights = [Uc_i / Uc for Uc_i in Uc_list]
    Us_total = sum(w * u_s for w, u_s in zip(weights, u_s_list))

    # Final Velocity Field
    U_total = U_in - Us_total

    # ---- Transverse velocities: V, W ----
    # (These remain the same)
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
    