import numpy as np

from .utils import VortexField

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
    gamma0 = config.gamma0
    merge_threshold = field_params.merge_threshold  # Distance threshold for vortex merging
    cfl_factor = field_params.cfl_factor       # CFL condition factor

    t = 0.0

    for index in range(total_steps):
        vor_field_list[index] = _vortex2velocity(vor_field_list[index], gamma0, config)
        # 1. Calculate effective turbulent viscosity
        Nu = _calculate_viscosity(dgamma, NuT_max)
        
        # 2. Compute induced velocities at all vortex locations
        V_induced, W_induced = _compute_mutual_induction(vor_field_list[index], gamma0)
        
        # 3. Adaptive time stepping
        dt = _calculate_time_step(vor_field_list[index], V_induced, W_induced, cfl_factor)
        t += dt
        
        # Check if simulation time exceeded
        if t * Uhub / D > 2 * field_params.max_X:
            break
            
        # 4. Update vortex positions and properties
        new_vor_field = _update_vortices(vor_field_list[index], V_induced, W_induced, dt, Nu)
        
        # 5. Merge close vortices
        new_vor_field = _merge_close_vortices(new_vor_field, merge_threshold)
        
        # 6. Store results
        new_vor_field.t = t
        vor_field_list.append(new_vor_field)
    
    return vor_field_list

def _vortex2velocity(data, gamma0, config):
    Y = data.Y
    Z = data.Z
    Circ = data.Circ
    Rv = data.Rv
    t = data.t
    yloc = config.yloc
    zloc = config.zloc
    V = np.zeros_like(yloc)
    W = np.zeros_like(yloc)

    for j in range(len(Circ)):
        Vj, Wj = _oseenlamb(Circ[j], Y[j], Z[j], Rv[j], yloc, zloc, gamma0, t)
        V += Vj
        W += Wj

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
        V=np.array([]),
        W=np.array([]),
        OmegaX=np.array([]),
        t=0.0
    )
    return [vordata] # return a list of vordata for extensibility

def _calculate_viscosity(dgamma, NuT_max):
    """Calculate effective turbulent viscosity based on negative circulation."""
    return -NuT_max * np.nansum(dgamma[dgamma < 0]) * 0.2

def _compute_mutual_induction(vortex_field, gamma0):
    """Compute induced velocities using Biot-Savart law with core correction."""
    Y, Z = vortex_field.Y, vortex_field.Z
    Circ, Rv = vortex_field.Circ, vortex_field.Rv
    t = vortex_field.t
    Nv = len(Y)
    
    # Create distance matrices
    Y_grid, Y_grid_T = np.meshgrid(Y, Y, indexing='ij')
    Z_grid, Z_grid_T = np.meshgrid(Z, Z, indexing='ij')
    Circ_grid = np.tile(Circ, (Nv, 1))
    Rv_grid = np.tile(Rv, (Nv, 1))
    
    dY = Y_grid - Y_grid_T
    dZ = Z_grid - Z_grid_T
    distance = np.sqrt(dY**2 + dZ**2)
    
    # Avoid division by zero and compute velocity magnitude
    with np.errstate(divide='ignore', invalid='ignore'):
        velocity_magnitude = (Circ_grid / (2 * np.pi * distance) * 
                            (1 - np.exp(-distance**2 / (0.02*gamma0*t + Rv_grid**2))))
        velocity_magnitude = np.nan_to_num(velocity_magnitude)
    
    # Compute velocity components
    with np.errstate(divide='ignore', invalid='ignore'):
        V = np.nan_to_num(-dZ / distance * velocity_magnitude)
        W = np.nan_to_num(dY / distance * velocity_magnitude)
    
    # Sum contributions from all vortices
    V_total = np.nansum(V, axis=1)
    W_total = np.nansum(W, axis=1)
    
    return V_total, W_total


def _calculate_time_step(vortex_field, V_induced, W_induced, cfl_factor):
    """Calculate adaptive time step based on CFL condition."""
    Y, Z = vortex_field.Y, vortex_field.Z
    Nv = len(Y)
    
    # Calculate minimum distance between vortices
    Y_grid, Y_grid_T = np.meshgrid(Y, Y, indexing='ij')
    Z_grid, Z_grid_T = np.meshgrid(Z, Z, indexing='ij')
    distance = np.sqrt((Y_grid - Y_grid_T)**2 + (Z_grid - Z_grid_T)**2)
    
    # Ignore diagonal (self-interaction)
    upper = distance[np.triu_indices(Nv, k=1)]
    positives = upper[upper > 0]
    min_distance = float(positives.min()) if positives.size else 1e-4

    # Maximum velocity magnitude
    max_velocity = np.max(np.abs(np.concatenate([V_induced, W_induced])))
    
    # CFL condition
    if max_velocity > 0:
        return min_distance / max_velocity * cfl_factor
    else:
        return 1e-4  # Fallback time step


def _update_vortices(vortex_field, V_induced, W_induced, dt, Nu):
    """Update vortex positions and core radii."""
    new_Y = vortex_field.Y + V_induced * dt
    new_Z = vortex_field.Z + W_induced * dt
    new_Rv = np.sqrt(vortex_field.Rv**2 + 4 * Nu * dt)
    
    return VortexField(
        Y=new_Y,
        Z=new_Z,
        Rv=new_Rv,
        Circ=vortex_field.Circ.copy(),
        yloc=vortex_field.yloc.copy(),
        zloc=vortex_field.zloc.copy(),
        V=vortex_field.V.copy(),
        W=vortex_field.W.copy(),
        OmegaX=vortex_field.OmegaX.copy(),
        t=vortex_field.t
    )

def _merge_close_vortices(vortex_field, threshold):
    """Merge vortices that are closer than the threshold distance."""
    Y, Z = vortex_field.Y, vortex_field.Z
    Circ, Rv = vortex_field.Circ, vortex_field.Rv
    
    # Calculate pairwise distances
    Y_grid, Y_grid_T = np.meshgrid(Y, Y, indexing='ij')
    Z_grid, Z_grid_T = np.meshgrid(Z, Z, indexing='ij')
    distance = np.sqrt((Y_grid - Y_grid_T)**2 + (Z_grid - Z_grid_T)**2)
    
    # Find pairs to merge (upper triangular to avoid duplicates)
    distance_upper = np.triu(distance, k=1)
    merge_pairs = np.where((distance_upper > 0) & (distance_upper <= threshold))
    
    # Mark vortices for removal
    to_remove = set()
    Y_new, Z_new, Circ_new, Rv_new = Y.copy(), Z.copy(), Circ.copy(), Rv.copy()
    
    for i, j in zip(merge_pairs[0], merge_pairs[1]):
        if i not in to_remove and j not in to_remove:
            # Merge vortex i into vortex j
            Circ_new[j] += Circ_new[i]
            Y_new[j] = 0.5 * (Y_new[i] + Y_new[j])
            Z_new[j] = 0.5 * (Z_new[i] + Z_new[j])
            # Keep the larger radius
            Rv_new[j] = max(Rv_new[i], Rv_new[j])
            to_remove.add(i)
    
    # Remove merged vortices
    if to_remove:
        keep_indices = [i for i in range(len(Y)) if i not in to_remove]
        Y_new = Y_new[keep_indices]
        Z_new = Z_new[keep_indices]
        Circ_new = Circ_new[keep_indices]
        Rv_new = Rv_new[keep_indices]
    
    return VortexField(
        Y=Y_new,
        Z=Z_new,
        Rv=Rv_new,
        Circ=Circ_new,
        yloc=vortex_field.yloc,
        zloc=vortex_field.zloc,
        V=vortex_field.V,
        W=vortex_field.W,
        OmegaX=vortex_field.OmegaX,
        t=vortex_field.t
    )

def _oseenlamb(Circ, Y, Z, Rv, yloc, zloc, gamma0, t):
    r = np.sqrt((yloc - Y) ** 2 + (zloc - Z) ** 2)
    # Avoid division by zero at vortex center
    r_safe = np.where(r == 0.0, 1.0, r)
    Ut = Circ / (2 * np.pi * r_safe) * (1 - np.exp(-r**2 / (0.02*gamma0*t + Rv**2)))
    # Ensure zero velocity at r==0
    Ut = np.where(r == 0.0, 0.0, Ut)
    V = - Ut * (zloc - Z) / r_safe
    W = Ut * (yloc - Y) / r_safe
    # Force zeros at exact center to avoid NaNs
    V = np.where(r == 0.0, 0.0, V)
    W = np.where(r == 0.0, 0.0, W)
    return V, W