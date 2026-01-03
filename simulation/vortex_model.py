import numpy as np

from .data_structures import VortexField

def simulate_vortex_evolution(config, field_params, total_steps=1000):
    """
    Simulate mutual induction of vortices in 2D plane with adaptive time stepping.
    
    Args:
        config: Simulation configuration
        field_params: Field parameters
        total_steps: Maximum number of time steps
        Tt: Maximum dimensionless time
    
    Returns:
        Updated vordata_list with evolved vortex states
    """
    vor_field_list = _define_location(config)

    # Extract parameters
    D = config.D
    Uhub = config.Uhub
    NuT_max = field_params.NuT_max
    dgamma = config.dgamma
    merge_threshold = field_params.merge_threshold  # Distance threshold for vortex merging
    cfl_factor = field_params.cfl_factor       # CFL condition factor

    t = 0.0

    for _ in range(total_steps):
        # Prepare data for induction calculations
        vor_field_list[-1] = _vortex2velocity(vor_field_list[-1], config)
        curr = vor_field_list[-1]
        dY, dZ, dist2 = _get_relative_geometry(curr.Y, curr.Z)

        # 1. Calculate effective turbulent viscosity
        Nu = _calculate_viscosity(dgamma, NuT_max)
        
        # 2. Compute induced velocities at all vortex locations
        Circ = curr.Circ
        Rv = curr.Rv
        V_induced, W_induced = _compute_mutual_induction(Circ, Rv, dY, dZ, dist2)
        
        # 3. Adaptive time stepping
        dt = _calculate_time_step(V_induced, W_induced, cfl_factor, dist2)
        t += dt
        
        # Check if simulation time exceeded
        if t * Uhub / D > 2 * field_params.max_X:
            break

        new_Y, new_Z = RK4_step(curr.Y, curr.Z, V_induced, W_induced, Circ, Rv, dt)
        new_Rv = np.sqrt(Rv**2 + 4 * Nu * dt)
            
        # 4. Update vortex positions and properties
        new_vor_field = VortexField(
            Y=new_Y, Z=new_Z, Rv=new_Rv, 
            Circ=curr.Circ.copy(), t=t,
            yloc=curr.yloc, zloc=curr.zloc, V=np.zeros_like(curr.V), W=np.zeros_like(curr.W), OmegaX=curr.OmegaX
        )
        
        # 5. Merge close vortices
        dY, dZ, dist2 = _get_relative_geometry(curr.Y, curr.Z)
        new_vor_field = _merge_close_vortices(new_vor_field, merge_threshold, dist2)
        
        # 6. Store results
        new_vor_field.t = t
        vor_field_list.append(new_vor_field)
    
    return vor_field_list

def _vortex2velocity(data, config):
    Y = data.Y
    Z = data.Z
    Circ = data.Circ
    Rv = data.Rv
    yloc = config.yloc
    zloc = config.zloc

    V, W = _oseenlamb(Circ, Y, Z, Rv, yloc, zloc)
    V += data.V
    W += data.W

    _, dVdZ = np.gradient(V, yloc[:, 0], zloc[0, :])
    dWdY, _ = np.gradient(W, yloc[:, 0], zloc[0, :])
    data.OmegaX = dWdY - dVdZ

    data.yloc = yloc
    data.zloc = zloc
    data.V = V
    data.W = W

    return data

def _define_location(config):
    # 1. Initial ring vortices
    Y = config.D / 2 * np.cos(config.phi) * np.cos(config.beta) - config.Yoffset
    Z = config.D / 2 * np.sin(config.phi) + config.Zhub
    Rv = np.full(config.Nv, 0.05 * config.D)
    Circ = config.dgamma.copy()

    # 2. Add hub vortex (center)
    Y = np.append(Y, -config.Yoffset)
    Z = np.append(Z, config.Zhub)
    Rv = np.append(Rv, 0.15 * config.D)
    Circ = np.append(Circ, -np.sum(config.dgamma))

    # 3. Mirror vortices against the wall (ground)
    Y = np.concatenate([Y, Y])
    Z = np.concatenate([Z, -Z])
    Rv = np.concatenate([Rv, Rv])
    Circ = np.concatenate([Circ, -Circ])

    # 4. Create VortexField object
    vordata = VortexField(
        Y=Y,
        Z=Z,
        Rv=Rv,
        Circ=Circ,
        yloc=np.array([]),
        zloc=np.array([]),
        V=config.V,
        W=config.W,
        OmegaX=np.array([]),
        t=0.0
    )
    return [vordata] # return a list of vordata for extensibility

def _calculate_viscosity(dgamma, NuT_max):
    """Calculate effective turbulent viscosity based on negative circulation."""
    return -NuT_max * np.nansum(dgamma[dgamma < 0]) * 0.2

def _compute_mutual_induction(Circ, Rv, dY, dZ, dist2):
    """Compute induced velocities using Biot-Savart law with core correction."""
    
    # Avoid division by zero and compute velocity magnitude
    with np.errstate(divide='ignore', invalid='ignore'):
        velocity_magnitude = (Circ[np.newaxis, :] / (2 * np.pi * dist2) * 
                            (1 - np.exp(-dist2 / (Rv[np.newaxis, :]**2))))
        velocity_magnitude = np.nan_to_num(velocity_magnitude)
    
        # Compute velocity components
        V = np.nan_to_num(-dZ * velocity_magnitude)
        W = np.nan_to_num(dY * velocity_magnitude)
    
    # Sum contributions from all vortices
    V_total = np.nansum(V, axis=1)
    W_total = np.nansum(W, axis=1)
    
    return V_total, W_total


def _calculate_time_step(V_induced, W_induced, cfl_factor, dist2):
    """Calculate adaptive time step based on CFL condition."""
    # Minimum distance between vortices
    min_distance = np.sqrt(np.min(dist2))

    # Maximum velocity magnitude
    max_velocity = np.max(np.abs(np.concatenate([V_induced, W_induced])))
    
    # CFL condition
    if max_velocity > 0:
        return min_distance / max_velocity * cfl_factor
    else:
        return 1e-4  # Fallback time step

def _merge_close_vortices(vortex_field, threshold, dist_matrix):
    """Merge vortices that are closer than the threshold distance."""
    Circ = vortex_field.Circ

    N_real = len(Circ) // 2
    Y_new = vortex_field.Y[:N_real].copy()
    Z_new = vortex_field.Z[:N_real].copy()
    Circ_new = vortex_field.Circ[:N_real].copy()
    Rv_new = vortex_field.Rv[:N_real].copy()

    real_dist_matrix = dist_matrix[:N_real, :N_real]

    rows, cols = np.where(real_dist_matrix <= threshold**2)
    unique = rows < cols
    merge_pairs = (rows[unique], cols[unique])
    
    # Mark vortices for removal
    to_remove = set()
    
    for i, j in zip(merge_pairs[0], merge_pairs[1]):
        if i not in to_remove and j not in to_remove:
            total_circ = Circ_new[i] + Circ_new[j]
            if abs(total_circ) > 1e-12:
                Y_new[j] = (Y_new[i] * Circ_new[i] + Y_new[j] * Circ_new[j]) / total_circ
                Z_new[j] = (Z_new[i] * Circ_new[i] + Z_new[j] * Circ_new[j]) / total_circ
                Rv_new[j] = np.sqrt((Circ_new[i] * Rv_new[i]**2 + Circ_new[j] * Rv_new[j]**2) / abs(total_circ))
                to_remove.add(i)
            else:
                # If total circulation is zero, remove both vortices
                to_remove.add(i)
                to_remove.add(j)
    
    # Remove merged vortices
    if to_remove:
        keep_indices = [i for i in range(N_real) if i not in to_remove]
        Y_new = Y_new[keep_indices]
        Z_new = Z_new[keep_indices]
        Circ_new = Circ_new[keep_indices]
        Rv_new = Rv_new[keep_indices]
    
    return VortexField(
        Y=np.concatenate([Y_new, Y_new]),
        Z=np.concatenate([Z_new, -Z_new]),
        Rv=np.concatenate([Rv_new, Rv_new]),
        Circ=np.concatenate([Circ_new, -Circ_new]),
        yloc=vortex_field.yloc,
        zloc=vortex_field.zloc,
        V=vortex_field.V,
        W=vortex_field.W,
        OmegaX=vortex_field.OmegaX,
        t=vortex_field.t
    )

def _oseenlamb(Circ, Y, Z, Rv, yloc, zloc):
    # Reshape vortex properties to (N, 1, 1) to enable broadcasting against (H, W)
    Y = Y[:, np.newaxis, np.newaxis]
    Z = Z[:, np.newaxis, np.newaxis]
    Circ = Circ[:, np.newaxis, np.newaxis]
    Rv = Rv[:, np.newaxis, np.newaxis]

    # Calculate relative distances: Shape (N, H, W)
    dY = yloc[np.newaxis, :, :] - Y
    dZ = zloc[np.newaxis, :, :] - Z
    r2 = dY**2 + dZ**2

    r2_safe = np.where(r2 == 0, 1e-18, r2)  # Prevent division by zero
    prefactor = (Circ / (2 * np.pi * r2_safe)) * (1 - np.exp(-r2 / Rv**2))

    V = np.sum(-prefactor * dZ, axis=0)
    W = np.sum(prefactor * dY, axis=0)
    return V, W

def RK4_step(Y, Z, V_k1, W_k1, Circ, Rv, dt):
    """Integrates positions using RK4 and enforces ground symmetry."""
    N_real = len(Circ) // 2
    
    # k2: Midpoint using k1
    Y2 = Y + V_k1 * (dt/2)
    Z2 = Z + W_k1 * (dt/2)
    Z2[N_real:] = -Z2[:N_real] # Enforce ground symmetry
    k2_v, k2_w = _get_vortex_derivatives(Y2, Z2, Circ, Rv)
    
    # k3: Midpoint using k2
    Y3 = Y + k2_v * (dt/2)
    Z3 = Z + k2_w * (dt/2)
    Z3[N_real:] = -Z3[:N_real] # Enforce ground symmetry
    k3_v, k3_w = _get_vortex_derivatives(Y3, Z3, Circ, Rv)
    
    # k4: End point using k3
    Y4 = Y + k3_v * dt
    Z4 = Z + k3_w * dt
    Z4[N_real:] = -Z4[:N_real] # Enforce ground symmetry
    k4_v, k4_w = _get_vortex_derivatives(Y4, Z4, Circ, Rv)
    
    # Final Position Update
    new_Y = Y + (dt/6.0) * (V_k1 + 2*k2_v + 2*k3_v + k4_v)
    new_Z = Z + (dt/6.0) * (W_k1 + 2*k2_w + 2*k3_w + k4_w)
    new_Z[N_real:] = -new_Z[:N_real] # Final symmetry enforcement
    
    return new_Y, new_Z

def _get_relative_geometry(Y, Z):
    """Compute relative geometry matrices for vortices."""
    # Create distance matrices
    Y_grid, Y_grid_T = np.meshgrid(Y, Y, indexing='ij')
    Z_grid, Z_grid_T = np.meshgrid(Z, Z, indexing='ij')
    
    dY = Y_grid - Y_grid_T
    dZ = Z_grid - Z_grid_T
    dist2 = dY**2 + dZ**2

    # Fill diagonal with inf to avoid division by zero
    np.fill_diagonal(dist2, np.inf)
    return dY, dZ, dist2

def _get_vortex_derivatives(Y, Z, Circ, Rv):
    """Calculates the velocity of each vortex core."""
    dY, dZ, dist2 = _get_relative_geometry(Y, Z)
    
    V_ind, W_ind = _compute_mutual_induction(Circ, Rv, dY, dZ, dist2)
    return V_ind, W_ind