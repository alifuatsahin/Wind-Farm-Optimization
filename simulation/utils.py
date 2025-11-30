from scipy.ndimage import uniform_filter, gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import os
from dataclasses import dataclass, field
from matplotlib.animation import PillowWriter

@dataclass
class VortexField:
    Y: np.ndarray = field(default_factory=lambda: np.array([]))      # Vortex Y positions
    Z: np.ndarray = field(default_factory=lambda: np.array([]))      # Vortex Z positions
    Rv: np.ndarray = field(default_factory=lambda: np.array([]))     # Vortex core radii
    Circ: np.ndarray = field(default_factory=lambda: np.array([]))   # Vortex circulations
    yloc: np.ndarray = field(default_factory=lambda: np.array([]))   # For velocity field grid (optional)
    zloc: np.ndarray = field(default_factory=lambda: np.array([]))   # For velocity field grid (optional)
    V: np.ndarray = field(default_factory=lambda: np.array([]))      # Velocity field (optional)
    W: np.ndarray = field(default_factory=lambda: np.array([]))      # Velocity field (optional)
    U: np.ndarray = field(default_factory=lambda: np.array([]))      # Streamwise velocity field (optional)
    OmegaX: np.ndarray = field(default_factory=lambda: np.array([])) # Vorticity field (optional)
    t: float = 0.0
    X: float = 0.0  # streamwise position

def NuT_model(x, config, field_params):
    """Compute the turbulent viscosity Nu_T based on the distance from the hub."""
    NuT_hat = min(field_params.NuT_max, x / (5 * config.D) * field_params.NuT_max)
    return config.a * config.Uhub * config.D * NuT_hat

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

def interpolate_vortex_fields(vortex_data_list, X):
    """Interpolate vortex field at position X from a list of VortexField objects."""

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

    return VortexField(
        Y=d0.Y,
        Z=d0.Z,
        Rv=d0.Rv,
        Circ=d0.Circ,
        yloc=d0.yloc,
        zloc=d0.zloc,
        V=V,
        W=W,
        U=U,
        OmegaX=d0.OmegaX,
        X=X
    )

def smooth_2d(U, kernel_size=3, method='gaussian'):
    """
    2D smoothing with options:
      - method='gaussian' : Gaussian blur (smooth, continuous look). sigma ~ kernel_size/2.
      - method='uniform'  : equivalent box filter (fast).
      - method='nanmean'  : box filter that ignores NaNs (safe for masked data).

    Returns an array same-shape as U. Keeps NaNs where no valid neighbors exist (for nanmean).
    """
    U = np.asarray(U, dtype=float)
    if method == 'uniform':
        return uniform_filter(U, size=kernel_size, mode='reflect')
    elif method == 'gaussian':
        sigma = max(0.5, kernel_size / 2.0)
        return gaussian_filter(U, sigma=sigma, mode='reflect')
    else:
        raise ValueError("method must be one of {'gaussian','uniform'}")

def plot_farm_deficit_map(wind_farm, x_resolution=100, y_resolution=50, save_path=None):
    """
    Calculates the combined superposed velocity deficit on an X-Y grid at hub height.
    Uses a simple sum-of-squares superposition model for visualization purposes.
    
    Args:
        wind_farm (WindFarm): The simulated WindFarm object.
        x_resolution (int): Number of points in the X-direction.
        y_resolution (int): Number of points in the Y-direction.
        save_path (str, optional): Path to save the generated plot. If None, the plot is shown instead.

    Returns:
        tuple: (X_grid, Y_grid, U_deficit_total_normalized)
    """
    
    if not wind_farm.turbines:
        return None, None, None

    T0 = wind_farm.turbines[0]
    D = T0.D
    Uhub_ref = wind_farm.field_params.Uhub

    # 1. Define Visualization Domain (X-Y Plane at Hub Height)
    x_coords = [t.pos[0] for t in wind_farm.turbines]
    y_coords = [t.pos[1] for t in wind_farm.turbines]

    max_X = wind_farm.field_params.max_X * D
    max_Y = wind_farm.field_params.max_Y * D
    
    x_min = min(x_coords) - max_X  # Start 1D upstream of first turbine
    x_max = max(x_coords) + max_X # End at max wake length of the last turbine
    
    y_min = min(y_coords) - max_Y
    y_max = max(y_coords) + max_Y

    X_vis = np.linspace(x_min, x_max, x_resolution)
    Y_vis = np.linspace(y_min, y_max, y_resolution)
    X_grid, Y_grid = np.meshgrid(X_vis, Y_vis, indexing='ij')
    
    U_deficit_total = np.zeros_like(X_grid)

    plt.close('all')

    if X_grid is None:
        print("No wake data found for plotting the wind farm wake.")
        return

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use pcolormesh for a heatmap of the deficit
    im = ax.pcolormesh(X_grid, Y_grid, U_deficit_total, cmap='seismic', shading='auto', 
                        vmin=0, vmax=np.nanmax(U_deficit_total)) # Deficit is positive
    
    # Add colorbar
    fig.colorbar(im, ax=ax, label='Normalized Velocity Deficit (-U / Uhub)')
    
    # Add turbine locations
    for i, t in enumerate(wind_farm.turbines):
        # Plot hub center
        ax.plot(t.pos[0], t.pos[1], 'o', color='r', markersize=5, alpha=0.7, label=f'T{i} Hub')
        
        # Draw a line representing the rotor diameter and yaw direction
        D = t.D
        yaw_rad = np.deg2rad(t.yaw)
        
        # Draw a line segment to represent the turbine and its yaw
        ax.plot([t.pos[0] - D/2 * np.sin(yaw_rad), t.pos[0] + D/2 * np.sin(yaw_rad)],
                [t.pos[1] - D/2 * np.cos(yaw_rad), t.pos[1] + D/2 * np.cos(yaw_rad)], 
                '-', color='black', linewidth=3, alpha=0.6)
        
        ax.text(t.pos[0], t.pos[1] + 0.6 * D, f'T{i}', color='r', ha='center', fontsize=10)

        grid_info = _get_grid_info(t.wake_field[0], t)
        history = _extract_streamwise_history(t.wake_field, t, grid_info)
    
    # Set limits and labels
    ax.set_xlabel('Streamwise Distance X (m)')
    ax.set_ylabel('Cross-stream Distance Y (m)')
    ax.set_title(f'Superposed Wind Farm Wake Map (Z={t.Zhub:.1f}m)')

    # 3. Save or Show
    if save_path is not None:
        save_path = os.path.join(save_path, f"WindFarm_Wake_Map.png")
        fig.savefig(save_path, dpi=300)
        print(f"Saved farm wake visualization to {save_path}")
    else:
        plt.show()
    
def plot_data(Data, params, pause_interval=0.1, quiver_samples=35, 
              show_streamwise=True, save_path=None, fps=10, dpi=150):
    """
    Main driver to visualize wake data. Coordinates setup, data prep, and animation loop.
    """
    plt.close('all')
    
    # 1. Prepare common data and grids
    grid_info = _get_grid_info(Data[0], params)
    wake_history = _extract_streamwise_history(Data, params, grid_info) if show_streamwise else None
    
    # 2. Setup Figure and Axes
    fig, axes_dict = _setup_layout(show_streamwise and wake_history is not None)
    
    # 3. Animation / Plotting Loop
    # Container for mutable state (colorbars) to avoid 'nonlocal' mess
    plot_state = {'cbar_vort': None, 'cbar_vel': None, 'cbar_wake': None}

    def update_frame(entry):
        _render_snapshot(axes_dict, plot_state, entry, params, grid_info, quiver_samples)
        if wake_history:
            _render_streamwise(axes_dict, plot_state, entry, params, grid_info, wake_history)

    # Execution
    if save_path:
        _save_animation(fig, Data, update_frame, save_path, params.yaw, fps, dpi, pause_interval)
    else:
        _show_live(fig, Data, update_frame, pause_interval)

    return fig, axes_dict

def _get_grid_info(data_snapshot, params):
    """Normalizes grids and finds indices for hub-height slicing."""
    yloc = np.asarray(data_snapshot.yloc) / params.D
    zloc = np.asarray(data_snapshot.zloc) / params.D
    
    # Find indices closest to Center/Hub
    z_target = params.Zhub / params.D
    y_center_idx = yloc.shape[0] // 2
    z_hub_idx = np.argmin(np.abs(zloc[0, :] - z_target))
    
    return {
        'yloc': yloc, 'zloc': zloc,
        'y_center_idx': y_center_idx, 'z_hub_idx': z_hub_idx,
        'Ny': yloc.shape[0], 'Nz': yloc.shape[1]
    }

def _extract_streamwise_history(Data, params, grid):
    """Pre-processes the full dataset to extract streamwise evolution arrays."""
    X_pos, U_hub_profiles = [], []
    
    for d in Data:
        X_pos.append(d.X / params.D)
        # Extract entire Y-profile at hub height Z
        U_hub_profiles.append(np.asarray(d.U)[:, grid['z_hub_idx']])
            
    if len(X_pos) < 2: return None

    # Create meshgrid for the bottom-right heatmap
    X_grid, Y_grid = np.meshgrid(X_pos, grid['yloc'][:, 0])
    
    return {
        'X': np.array(X_pos),
        'U_hub_2D': np.array(U_hub_profiles).T, # Shape: (Ny, N_snapshots)
        'X_grid': X_grid,
        'Y_grid': Y_grid
    }

def _setup_layout(has_streamwise):
    """Creates the figure and returns a labeled dictionary of axes."""
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    
    if has_streamwise:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        return fig, {'vort': axes[0,0], 'vel': axes[0,1], 'prof': axes[1,0], 'wake': axes[1,1]}
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        return fig, {'vort': axes[0], 'vel': axes[1]}

def _render_snapshot(axes, state, entry, params, grid, q_samples):
    """Handles the top row: Contour plots for current timestep."""
    # 1. Vorticity (Top Left)
    ax = axes['vort']
    ax.clear()
    ax.set_aspect('equal', 'box')
    
    omg = np.asarray(entry.OmegaX)
    norm_factor = params.Uhub / params.D
    omg_norm = omg / norm_factor
    
    limit = np.nanmax(np.abs(omg_norm)) if np.isfinite(np.nanmax(omg_norm)) else 1.0
    levels = np.linspace(-1, 1, 21) * limit
    
    cf = ax.contourf(grid['yloc'], grid['zloc'], omg_norm, levels=levels, cmap='RdBu_r', extend='both')
    ax.set_title(f'$\\Omega_x$ (Normalized) at X/D = {entry.X/params.D:.1f}')
    
    if state['cbar_vort'] is None:
        state['cbar_vort'] = ax.figure.colorbar(cf, ax=ax, label='Vorticity')
    else:
        state['cbar_vort'].update_normal(cf)
        
    # Quiver overlay
    _add_quiver(ax, entry, grid, q_samples)

    # 2. Velocity (Top Right)
    ax = axes['vel']
    ax.clear()
    ax.set_aspect('equal', 'box')

    u_def_norm = np.asarray(params.Uin - entry.U) / params.Uhub
    limit = np.nanmax(np.abs(u_def_norm)) if np.isfinite(np.nanmax(u_def_norm)) else 1.0

    cf = ax.contourf(grid['yloc'], grid['zloc'], u_def_norm, levels=21, cmap='turbo')
    cf.set_clim(0, limit)
    ax.set_title(f'U/Uhub at X/D = {entry.X/params.D:.1f}')
    
    if state['cbar_vel'] is None:
        state['cbar_vel'] = ax.figure.colorbar(cf, ax=ax, label='Normalized Streamwise Velocity')
    else:
        state['cbar_vel'].update_normal(cf)

def _add_quiver(ax, entry, grid, samples):
    """Adds quiver arrows to an existing axis."""
    dN = max(1, int(grid['Ny'] / samples))
    sl = np.s_[::dN, ::dN] # Slice object
    ax.quiver(grid['yloc'][sl], grid['zloc'][sl], 
              np.asarray(entry.V)[sl], np.asarray(entry.W)[sl], 
              color='k', scale=16.0, angles='xy')

def _render_streamwise(axes, state, entry, params, grid, history):
    """Handles bottom row: Profiles and Heatmap."""
    current_x = entry.X / params.D
    
    # 3. Radial Profiles (Bottom Left)
    ax = axes['prof']
    ax.clear()
    
    # Plot background profiles (static snapshots)
    num_profs = min(5, len(history['X']))
    indices = np.linspace(0, len(history['X'])-1, num_profs, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, num_profs))
    
    for i, idx in enumerate(indices):
        ax.plot(grid['yloc'][:,0], history['U_hub_2D'][:, idx], 
                color=colors[i], marker='o', ms=3, alpha=0.5, label=f'X/D={history["X"][idx]:.1f}')
        
    # Highlight current profile
    curr_idx = np.argmin(np.abs(history['X'] - current_x))
    ax.plot(grid['yloc'][:,0], history['U_hub_2D'][:, curr_idx], 'r-', lw=3, label='Current')
    
    ax.set_title('Hub Height Velocity Profiles')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 4. Wake Evolution Heatmap (Bottom Right)
    ax = axes['wake']
    ax.clear()
    
    u_norm = -history['U_hub_2D'] / params.Uhub
    mesh = ax.pcolormesh(history['X_grid'], history['Y_grid'], u_norm, cmap='jet', shading='auto')
    ax.axvline(current_x, color='r', ls='--')
    ax.set_title('Wake Evolution (Top Down)')
    
    if state['cbar_wake'] is None:
        state['cbar_wake'] = ax.figure.colorbar(mesh, ax=ax, label='Deficit')
    else:
        state['cbar_wake'].update_normal(mesh)

def _save_animation(fig, data, update_func, path, yaw, fps, dpi, interval):
    """Handles the video writing logic."""
    try:
        out_path = os.path.join(path, f"Yaw{yaw:.2f}_Wake.gif")
        writer = PillowWriter(fps=fps)
        with writer.saving(fig, out_path, dpi):
            for entry in data:
                update_func(entry)
                fig.canvas.draw()
                writer.grab_frame()
                plt.pause(interval) # Optional: keep small pause to see progress
    except Exception as e:
        print(f"Error saving animation: {e}")

def _show_live(fig, data, update_func, interval):
    """Handles standard live plotting."""
    for entry in data:
        update_func(entry)
        plt.pause(interval)