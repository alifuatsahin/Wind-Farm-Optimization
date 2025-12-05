from scipy.ndimage import uniform_filter, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.patheffects as patheffects
import numpy as np
import os

from .superposition import superpose, interpolate_local_velocity_field

def NuT_model(x, config, field_params):
    """Compute the turbulent viscosity Nu_T based on the distance from the hub."""
    NuT_hat = min(field_params.NuT_max, x / (5 * config.D) * field_params.NuT_max)
    return config.a * config._init_Uhub() ** 2 / config.Uhub * config.D * NuT_hat

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

def plot_farm_deficit_map(wind_farm, x_resolution=300, y_resolution=100, z_resolution=100, save_path=None):
    if not wind_farm.turbines:
        print("No turbines found in wind farm.")
        return

    print("Generating Momentum Conserving Superposition Map...")

    # 1. Define Visualization Domain
    all_x = [t.pos[0] for t in wind_farm.turbines]
    all_y = [t.pos[1] for t in wind_farm.turbines]
    D = wind_farm.turbines[0].D 
    Uhub_ref = wind_farm.turbines[0].Uhub 
    Zhub_ref = wind_farm.turbines[0].Zhub 

    max_wake_len = wind_farm.field_params.max_X * D
    max_wake_wid = wind_farm.field_params.max_Y * D
    max_wake_hgt = wind_farm.field_params.max_Z * D
    
    x_min = min(all_x) - 2 * D
    x_max = max(all_x) + max_wake_len
    y_min = min(all_y) - max_wake_wid
    y_max = max(all_y) + max_wake_wid
    z_min = 0.0
    z_max = max([t.Zhub for t in wind_farm.turbines]) + max_wake_hgt

    X_vis = np.linspace(x_min, x_max, x_resolution)
    Y_vis = np.linspace(y_min, y_max, y_resolution)
    Z_vis = np.linspace(z_min, z_max, z_resolution)

    Y_loc, Z_loc = np.meshgrid(Y_vis, Z_vis, indexing='ij')

    # 2. Calculate Background Flow
    fp = wind_farm.field_params
    z_safe = np.maximum(Z_loc, fp.z0 + 1e-3)
    U_in = fp.Uh * (np.log(z_safe / fp.z0) / np.log(fp.Zh / fp.z0))
    
    z_ref_idx = np.argmin(np.abs(Z_vis - Zhub_ref))
    y_ref_idx = np.argmin(np.abs(Y_vis - 0.0)) 

    # Initialize Maps
    U_xy_map = np.zeros((len(Y_vis), len(X_vis))) # (Y, X)
    U_xz_map = np.zeros((len(Z_vis), len(X_vis))) # (Z, X)

    # 3. Streamwise Iteration
    for i, x_global in enumerate(X_vis):
        
        U_wake_list = []
        
        for t in wind_farm.turbines:
            dist = x_global - t.pos[0]
            
            if dist > 0 and dist < wind_farm.field_params.max_X * t.D:
                # Interpolate returns a (Y, Z) slice
                u_local_abs = interpolate_local_velocity_field(
                    t, dist, Y_loc, Z_loc, default=U_in
                )
                U_wake_list.append(u_local_abs)

        if len(U_wake_list) > 0:
            # Pass 2D background and 2D wake list
            U_total_slice = superpose(U_in, np.array(U_wake_list), method='RSS')[0]
        else:
            U_total_slice = U_in

        # Top View: Take all Y at fixed Z (Reference Hub Height)
        U_xy_map[:, i] = U_total_slice[:, z_ref_idx]
        
        # Side View: Take all Z at fixed Y (Centerline)
        U_xz_map[:, i] = U_total_slice[y_ref_idx, :]

    U_xy_map = smooth_2d(U_xy_map, method='gaussian')
    U_xz_map = smooth_2d(U_xz_map, method='gaussian')

    # 4. Plotting
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8)) # Added figsize
    
    # XY Grid
    X_grid_xy, Y_grid_xy = np.meshgrid(X_vis, Y_vis) 
    
    im1 = ax1.pcolormesh(X_grid_xy, Y_grid_xy, U_xy_map / Uhub_ref, 
                       cmap='bwr', shading='auto')
    
    # XZ Grid (Side view requires X and Z)
    X_grid_xz, Z_grid_xz = np.meshgrid(X_vis, Z_vis)
    
    im2 = ax2.pcolormesh(X_grid_xz, Z_grid_xz, U_xz_map / Uhub_ref, 
                       cmap='bwr', shading='auto')
        
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label(r'Normalized Velocity $U / U_{hub}$')
    ax1.set_xlabel('Global Streamwise X (m)')
    ax1.set_ylabel('Global Cross-stream Y (m)')
    ax1.set_title(f'Top View (Z = {Zhub_ref:.1f}m)')

    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label(r'Normalized Velocity $U / U_{hub}$')
    ax2.set_xlabel('Global Streamwise X (m)')
    ax2.set_ylabel('Global Vertical Z (m)')
    ax2.set_title('Side View (Y = 0m)')
    
    # --- TURBINE OVERLAYS ---
    for idx, t in enumerate(wind_farm.turbines):
        yaw_rad = np.deg2rad(t.yaw)
        
        # --- TOP VIEW (XY) ---
        # Draw a line representing the rotor diameter, rotated by yaw
        dx = (t.D / 2) * np.sin(yaw_rad)
        dy = (t.D / 2) * np.cos(yaw_rad)
        
        ax1.plot([t.pos[0] - dx, t.pos[0] + dx], 
                 [t.pos[1] + dy, t.pos[1] - dy], 
                 color='black', lw=3, solid_capstyle='round')
        
        ax1.text(t.pos[0], t.pos[1] + t.D * 0.6, f"T{idx}", color='white', 
                 ha='center', va='center', fontweight='bold',
                 path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])
        
        # --- SIDE VIEW (XZ) ---
        # 1. Draw the Tower (Vertical line from ground to hub)
        ax2.plot([t.pos[0], t.pos[0]], 
                 [0, t.Zhub], 
                 color='black', lw=2)
        
        # 2. Draw the Rotor (Vertical line at hub height)
        z_top = t.Zhub + (t.D / 2)
        z_bot = t.Zhub - (t.D / 2)
        
        ax2.plot([t.pos[0], t.pos[0]], 
                 [z_bot, z_top], 
                 color='black', lw=4, solid_capstyle='round')

    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved farm map to {save_path}")
    else:
        plt.show()
    
def plot_data(Data, config, pause_interval=0.1, quiver_samples=35,
              show_streamwise=True, save_path=None, fps=10, dpi=150):
    """
    Main driver to visualize wake data. Coordinates setup, data prep, and animation loop.
    """
    plt.close('all')
    
    # 1. Prepare common data and grids
    grid_info = _get_grid_info(Data[0], config, config.Zhub)
    wake_history = _extract_streamwise_history(Data, config, grid_info) if show_streamwise else None
    
    # 2. Setup Figure and Axes
    fig, axes_dict = _setup_layout(show_streamwise and wake_history is not None)
    
    # 3. Animation / Plotting Loop
    # Container for mutable state (colorbars) to avoid 'nonlocal' mess
    plot_state = {'cbar_vort': None, 'cbar_vel': None, 'cbar_wake': None}

    def update_frame(entry):
        _render_snapshot(axes_dict, plot_state, entry, config, grid_info, quiver_samples)
        if wake_history:
            _render_streamwise(axes_dict, plot_state, entry, config, grid_info, wake_history)

    # Execution
    if save_path:
        _save_animation(fig, Data, update_frame, save_path, config.yaw, fps, dpi, pause_interval)
    else:
        _show_live(fig, Data, update_frame, pause_interval)

    return fig, axes_dict

def _get_grid_info(data_snapshot, config, z_target):
    """Normalizes grids and finds indices for hub-height slicing."""
    yloc = np.asarray(data_snapshot.yloc) / config.D
    zloc = np.asarray(data_snapshot.zloc) / config.D
    
    # Find indices closest to Center/Hub
    z_target = z_target / config.D
    y_center_idx = yloc.shape[0] // 2
    z_hub_idx = np.argmin(np.abs(zloc[0, :] - z_target))
    
    return {
        'yloc': yloc, 'zloc': zloc,
        'y_center_idx': y_center_idx, 'z_hub_idx': z_hub_idx,
        'Ny': yloc.shape[0], 'Nz': yloc.shape[1]
    }

def _extract_streamwise_history(Data, config, grid):
    """Pre-processes the full dataset to extract streamwise evolution arrays."""
    X_pos, U_hub_profiles = [], []
    
    for d in Data:
        X_pos.append(d.X / config.D)
        # Extract entire Y-profile at hub height Z
        U_hub_profiles.append(np.asarray(d.U)[:, grid['z_hub_idx']])
            
    if len(X_pos) < 2: return None

    # Create meshgrid for the bottom-right heatmap
    X_grid, Y_grid = np.meshgrid(X_pos, grid['yloc'][:, 0])
    
    return {
        'X': np.array(X_pos),
        'U': np.array(U_hub_profiles).T, # Shape: (Ny, N_snapshots)
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

def _render_snapshot(axes, state, entry, config, grid, q_samples):
    """Handles the top row: Contour plots for current timestep."""
    # 1. Vorticity (Top Left)
    ax = axes['vort']
    ax.clear()
    ax.set_aspect('equal', 'box')
    
    omg = np.asarray(entry.OmegaX)
    norm_factor = config.Uhub / config.D
    omg_norm = omg / norm_factor
    
    limit = np.nanmax(np.abs(omg_norm)) if np.isfinite(np.nanmax(omg_norm)) else 1.0
    levels = np.linspace(-1, 1, 21) * limit
    
    cf = ax.contourf(grid['yloc'], grid['zloc'], omg_norm, levels=levels, cmap='RdBu_r', extend='both')
    ax.set_title(f'$\\Omega_x$ (Normalized) at X/D = {entry.X/config.D:.1f}')
    
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

    u_def_norm = np.asarray(entry.U) / config.Uhub
    limit = np.nanmax(np.abs(u_def_norm)) if np.isfinite(np.nanmax(u_def_norm)) else 1.0

    cf = ax.contourf(grid['yloc'], grid['zloc'], u_def_norm, levels=21, cmap='turbo')
    cf.set_clim(0, limit)
    ax.set_title(f'U/Uhub at X/D = {entry.X/config.D:.1f}')
    
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

def _render_streamwise(axes, state, entry, config, grid, history):
    """Handles bottom row: Profiles and Heatmap."""
    current_x = entry.X / config.D
    
    # 3. Radial Profiles (Bottom Left)
    ax = axes['prof']
    ax.clear()
    
    # Plot background profiles (static snapshots)
    num_profs = min(5, len(history['X']))
    indices = np.linspace(0, len(history['X'])-1, num_profs, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, num_profs))
    
    u_norm = history['U'] / config._init_Uhub()
    for i, idx in enumerate(indices):
        ax.plot(grid['yloc'][:,0], u_norm[:, idx], 
                color=colors[i], marker='o', ms=3, alpha=0.5, label=f'X/D={history["X"][idx]:.1f}')
        
    # Highlight current profile
    curr_idx = np.argmin(np.abs(history['X'] - current_x))
    ax.plot(grid['yloc'][:,0], u_norm[:, curr_idx], 'r-', lw=3, label='Current')
    
    ax.set_title('Hub Height Velocity Profiles')
    ax.set_ylabel('U/Uhub')
    ax.set_xlabel('Normalized Cross-stream Distance Y/D')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 4. Wake Evolution Heatmap (Bottom Right)
    ax = axes['wake']
    ax.clear()
    
    mesh = ax.pcolormesh(history['X_grid'], history['Y_grid'], u_norm, cmap='jet', shading='auto')
    ax.axvline(current_x, color='r', ls='--')
    ax.set_title('Wake Evolution (Top Down)')
    ax.set_xlabel('Streamwise Distance X/D')
    ax.set_ylabel('Normalized Cross-stream Distance Y/D')
    
    if state['cbar_wake'] is None:
        state['cbar_wake'] = ax.figure.colorbar(mesh, ax=ax, label='Normalized Velocity U/Uhub')
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