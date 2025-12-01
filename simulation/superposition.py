import numpy as np

from .utils import interpolate_vortex_fields

# def get_superposed_velocity_field(field_params, vortex_field, wind_farm):
#     """Compute the superposed velocity field at all downstream planes."""

#     upstream_turbines = [t for t in wind_farm.turbines if t.pos[0] < vortex_field.X]

#     if not upstream_turbines:
#         return vortex_field

#     wake_fields = [interpolate_vortex_fields(t.wake_field, vortex_field.X) 
#                    for t in upstream_turbines]
#     wake_fields.append(vortex_field)  # include self field

#     U_list = [field_params.Uhub - wf.U for wf in wake_fields]
#     V_list = [wf.V for wf in wake_fields]
#     W_list = [wf.W for wf in wake_fields]

#     U_wake, V_wake, W_wake, _ = momentum_conserving_superposition(
#         U_list, V_list, W_list, U_inf=field_params.Uinf
#     )

#     vortex_field.U = U_wake
#     vortex_field.V = V_wake
#     vortex_field.W = W_wake

#     return vortex_field

def get_local_velocity_field(config, wind_farm):
    """Compute the combined velocity field at a given downstream turbine plane."""

    upstream_turbines = [t for t in wind_farm.turbines if t.pos[0] < config.pos[0]]

    # Freestream inflow at that turbine plane
    U_base = config.Uin
    V_base = np.zeros_like(U_base)
    W_base = np.zeros_like(U_base)

    if not upstream_turbines:
        return U_base, V_base, W_base

    # Get wake fields at this x-plane
    wake_fields = [interpolate_vortex_fields(t.wake_field, config.pos[0]) 
                   for t in upstream_turbines]

    U_list = [wf.U for wf in wake_fields]
    V_list = [wf.V for wf in wake_fields]
    W_list = [wf.W for wf in wake_fields]

    # Momentum-conserving superposition
    U_wake, V_wake, W_wake, _ = momentum_conserving_superposition(
        U_list, V_list, W_list, U_in=config.Uin
    )

    # Total local wind = freestream + wake-induced perturbation
    U_total = U_wake
    V_total = V_wake
    W_total = W_wake

    return U_total, V_total, W_total

def momentum_conserving_superposition(U_list, V_list, W_list, U_in, max_iter=10, tol=1e-3):
    """
    Momentum-conserving superposition for U, V, W at a single x-plane.

    Inputs:
        U_list: list of 2D arrays (U_i of each turbine at this x-plane)
        V_list: list of 2D arrays (V_i of each turbine at this x-plane)
        W_list: list of 2D arrays (W_i of each turbine at this x-plane)
        U_in : free-stream velocity
    Returns:
        U_total, V_total, W_total, Uc (combined convection velocity)
    """

    # Convert to arrays
    U_list = [np.asarray(U) for U in U_list]
    V_list = [np.asarray(V) for V in V_list]
    W_list = [np.asarray(W) for W in W_list]

    # Individual velocity deficits u_i_s = U_in - U
    u_s_list = [U_in - U for U in U_list]

    # --- Initial convection velocity estimate: Uc_i = mean(U_i) ---
    Uc_list = [np.mean(U) for U in U_list]

    # Initial combined convection velocity: max(Uc_i)
    Uc = np.max(Uc_list)

    # ---- ITERATION LOOP (eq 2.7 & 2.9) ----
    for _ in range(max_iter):

        # eq 2.9: weighted deficit
        weights = [Uc_i / Uc for Uc_i in Uc_list]
        Us_total = sum(w * u_s for w, u_s in zip(weights, u_s_list))

        # eq 2.7: update Uc = ∫ Uw * Us / ∫ Us
        Uw_total = U_in - Us_total  # reminder: Us = U_inf - U
        num = np.sum(Uw_total * Us_total)
        den = np.sum(Us_total)

        if den == 0:
            break

        Uc_new = num / den

        if np.abs(Uc_new - Uc) / np.abs(Uc_new) < tol:
            Uc = Uc_new
            break

        Uc = Uc_new

    # Recalculate the final weighted deficit (Uw_total_final)
    weights = [Uc_i / Uc for Uc_i in Uc_list]
    Us_total = sum(w * u_s for w, u_s in zip(weights, u_s_list))
    U_total = U_in - Us_total

    # ---- Transverse velocities: V, W ----
    # Use same weights as eq 4.1:   V = Σ (Uc_i / Uc) * V_i
    V_total = sum((Uc_i / Uc) * V for Uc_i, V in zip(Uc_list, V_list))
    W_total = sum((Uc_i / Uc) * W for Uc_i, W in zip(Uc_list, W_list))

    return U_total, V_total, W_total, Uc
    