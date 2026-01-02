import numpy as np

def advance_wake_field(data, dt, NuT, config, field_params):
    """Update the streamwise velocity according to transportation-diffusion equations."""
    Uin = config.Uin
    U = data.U.copy()
    V = data.V.copy()
    W = data.W.copy()
    yloc = data.yloc
    zloc = data.zloc
 
    V_veer = U * np.sin(np.deg2rad(field_params.WV * (zloc - config.Zhub)))
    V += V_veer # total transverse velocity

    # estimate mean convection distance
    Us = Uin - U
    mask = Us > 0.2 * np.max(Us)
    dx = dt * np.mean(U[mask])

    # prediction step
    dUdY, dUdZ = np.gradient(U, yloc[:, 0], zloc[0, :])   # dUdY = ∂(U)/∂y, dUdZ = ∂(U)/∂z
    # Gradient of Uin (Background Shear)
    _, dUin_dZ = np.gradient(Uin, yloc[:, 0], zloc[0, :], axis=(0, 1))

    Sxy = dUdY
    Sxz = dUdZ

    # derivatives of strains:
    dSxydY, _ = np.gradient(Sxy, yloc[:, 0], zloc[0, :])   # first output = ∂Sxy/∂y
    _, dSxzdZ = np.gradient(Sxz, yloc[:, 0], zloc[0, :])   # second output = ∂Sxz/∂z
    _, d2Uin_dZ2 = np.gradient(dUin_dZ, yloc[:, 0], zloc[0, :], axis=(0, 1))

    # assemble numerator
    numer = -V * dUdY - W * dUdZ + NuT * dSxydY + NuT * dSxzdZ - NuT * d2Uin_dZ2
    denom_safe = np.maximum(U, 1e-6)  # prevent division by zero

    dUpdx = numer / denom_safe
    Up = U + dUpdx * dx
    Up[:, 0] = Uin[:, 0] # far stream dirichlet BC

    # correction step
    dUdY, dUdZ = np.gradient(Up, yloc[:, 0], zloc[0, :])   # dUdY = ∂(Up)/∂y, dUdZ = ∂(Up)/∂z
    Sxz = dUdZ
    Sxy = dUdY

    dSxydY, _ = np.gradient(Sxy, yloc[:, 0], zloc[0, :])   # first output = ∂Sxy/∂y
    _, dSxzdZ = np.gradient(Sxz, yloc[:, 0], zloc[0, :])   # second output = ∂Sxz/∂z
    numer = -V * dUdY - W * dUdZ + NuT * dSxydY + NuT * dSxzdZ - NuT * d2Uin_dZ2
    denom_safe = np.maximum(Up, 1e-6)  # prevent division by zero

    dUcdx = numer / denom_safe
    U += 0.5 * (dUpdx + dUcdx) * dx
    U[:, 0] = Uin[:, 0] # far stream dirichlet BC
    
    return U, data.X + dx