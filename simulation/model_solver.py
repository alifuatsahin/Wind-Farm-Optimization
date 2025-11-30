import numpy as np

def advance_wake_field(data, dt, NuT, config, field_params):
    """Update the streamwise velocity according to transportation-diffusion equations."""
    Uin = config.Uin
    Uw = data.U.copy()  # Make a copy to avoid modifying original
    V = data.V.copy()
    W = data.W.copy()
    yloc = data.yloc
    zloc = data.zloc
 
    V_veer = Uin * np.sin(np.deg2rad(field_params.WV * (zloc - config.Zhub)))
    V += V_veer # total transverse velocity

    # estimate mean convection distance
    mask = Uw > 0.2 * np.max(Uw)
    mean_Uw = np.mean(Uw[mask]) if np.any(mask) else config.Uhub
    dx = dt * (np.mean(Uin) - mean_Uw)

    # prediction step
    dUdY, dUdZ = np.gradient(Uw, yloc[:, 0], zloc[0, :])   # dUdY = ∂(Uw)/∂y, dUdZ = ∂(Uw)/∂z
    _, dUin_dZ = np.gradient(Uin, yloc[:, 0], zloc[0, :])   # dUin_dZ = ∂(Uin)/∂z

    Sxy = dUdY
    Sxz = dUdZ

    # derivatives of strains:
    dSxydY, _ = np.gradient(Sxy, yloc[:, 0], zloc[0, :])   # first output = ∂Sxy/∂y
    _, dSxzdZ = np.gradient(Sxz, yloc[:, 0], zloc[0, :])   # second output = ∂Sxz/∂z

    # assemble numerator
    numer = -V * dUdY + W * (-dUdZ + dUin_dZ) + NuT * dSxydY + NuT * dSxzdZ
    denom = Uin - Uw

    dUwpdx = numer / denom
    Uwp = Uw + dUwpdx * dx
    Uwp[:, 0] = 0.0 # no slip at BC

    # correction step
    dUdY, dUdZ = np.gradient(Uwp, yloc[:, 0], zloc[0, :])   # dUdY = ∂(Uwp)/∂y, dUdZ = ∂(Uwp)/∂z

    Sxz = dUdZ
    Sxy = dUdY

    dSxydY, _ = np.gradient(Sxy, yloc[:, 0], zloc[0, :])   # first output = ∂Sxy/∂y
    _, dSxzdZ = np.gradient(Sxz, yloc[:, 0], zloc[0, :])   # second output = ∂Sxz/∂z
    numer = -V * dUdY + W * (-dUdZ + dUin_dZ) + NuT * dSxydY + NuT * dSxzdZ
    denom = Uin - Uwp

    dUwcdx = numer / denom
    Uw += 0.5 * (dUwpdx + dUwcdx) * dx
    Uw[:, 0] = 0.0 # no slip at BC
    
    return Uw, data.X + dx