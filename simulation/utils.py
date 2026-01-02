from scipy.ndimage import uniform_filter, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.patheffects as patheffects
import numpy as np
import os

from .superposition import superpose, interpolate_local_velocity_field

def NuT_model(wake_field, config, upstream_turbines):
    """Compute the turbulent viscosity Nu_T based on the distance from the hub."""

    I_amb = 0.07  # ambient turbulence intensity
    m = 2.0
    I = I_amb ** m # initialize as ambient turbulence intensity
    delta_I = lambda turbine, x_val: 0.73 * turbine.a ** 0.8325 * I_amb ** (-0.03) * (x_val / turbine.D) ** -0.32

    if wake_field.X > 5 * config.D:
        I += delta_I(config, wake_field.X) ** m
    else:
        I += delta_I(config, 5 * config.D) ** m

    for t in upstream_turbines:
        dist = wake_field.X + (config.pos[0] - t.pos[0])
        I_add = delta_I(t, dist)
        I += I_add ** m

    I_total = I ** (1/m) / 0.1489 * 1.5

    NuT_hat = min(0.03 * (1 - (wake_field.X - 5 * config.D) / (100 * config.D)), wake_field.X / (5 * config.D) * 0.06)
    NuT_hat = config.a * config._init_Uhub() * config.D * NuT_hat * I_total

    for t in upstream_turbines:
        dist = wake_field.X + (config.pos[0] - t.pos[0])
        NuT_add = min(0.03 * (1 - (dist - 5 * t.D) / (100 * t.D)), dist / (5 * t.D) * 0.06)
        NuT_hat += NuT_add * t.a * t._init_Uhub() * t.D * I_total

    # I_amb = 0.07  # ambient turbulence intensity
    # delta_I = lambda turbine, x_val: 0.73 * turbine.a ** 0.8325 * I_amb ** (-0.03) * (x_val / turbine.D) ** -0.32
    
    # nu_t = lambda turbine, I_total: 0.015 * turbine.Uinf * turbine.D * I_total
    # I = I_amb ** 2 # initialize as ambient turbulence intensity

    # if wake_field.X > 5 * config.D:
    #     I += delta_I(config, wake_field.X) ** 2
    # else:
    #     I += delta_I(config, 5 * config.D) ** 2

    # for t in upstream_turbines:
    #     dist = wake_field.X + (config.pos[0] - t.pos[0])
    #     I_add = delta_I(t, dist)
    #     I += I_add ** 2

    # I_total = np.sqrt(I)
    # NuT_hat = nu_t(config, I_total)

    return NuT_hat

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

    elevation_func = wind_farm.turbine_configs.elevation_func

    # 1. Define Visualization Domain
    all_x = [t.pos[0] for t in wind_farm.turbines]
    all_y = [t.pos[1] for t in wind_farm.turbines]
    D = wind_farm.turbines[0].D 
    Uhub_ref = wind_farm.turbines[0].Uhub 
    Zhub_ref = wind_farm.turbines[0].Zhub + wind_farm.turbines[0].pos[2]
    y_slice_val = wind_farm.turbines[0].pos[1]  # Cross-stream slice at T0 for side view
    z_min_plot = 0.0  # Minimum z for plotting ground

    max_wake_len = wind_farm.field_params.max_X * D
    max_wake_wid = wind_farm.field_params.max_Y * D
    max_wake_hgt = wind_farm.field_params.max_Z * D
    
    x_min = min(all_x) - 2 * D
    x_max = max(all_x) + 5 * D
    y_min = min(all_y) - max_wake_wid
    y_max = max(all_y) + max_wake_wid
    z_min = 0.0
    z_max = max([(t.Zhub + t.pos[2]) for t in wind_farm.turbines]) + max_wake_hgt

    X_vis = np.linspace(x_min, x_max, x_resolution)
    Y_vis = np.linspace(y_min, y_max, y_resolution)
    Z_vis = np.linspace(z_min, z_max, z_resolution)

    Y_loc, Z_loc = np.meshgrid(Y_vis, Z_vis, indexing='ij')

    # 2. Calculate Background Flow
    fp = wind_farm.field_params
    z_safe = np.maximum(Z_loc, fp.z0 + 1e-3)
    U_in = fp.Uh * (np.log(z_safe / fp.z0) / np.log(fp.Zh / fp.z0))
    
    z_ref_idx = np.argmin(np.abs(Z_vis - Zhub_ref))
    y_ref_idx = np.argmin(np.abs(Y_vis - y_slice_val)) 

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
                 [t.pos[2], t.Zhub+t.pos[2]], 
                 color='black', lw=2)
        
        # 2. Draw the Rotor (Vertical line at hub height)
        z_top = t.pos[2] + t.Zhub + (t.D / 2)
        z_bot = t.pos[2] + t.Zhub - (t.D / 2)
        
        ax2.plot([t.pos[0], t.pos[0]], 
                 [z_bot, z_top], 
                 color='black', lw=4, solid_capstyle='round')

    # --- DRAW GROUND IN SIDE VIEW ---
    if elevation_func:
        # Calculate terrain line along the side-view slice
        z_ground_line = elevation_func(X_vis, np.full_like(X_vis, y_slice_val))
        ax2.fill_between(X_vis, z_min_plot, z_ground_line, color='#6d4c41', alpha=1.0, zorder=5) # Brown ground
        ax2.plot(X_vis, z_ground_line, color='black', lw=1, zorder=6)

    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        grid = wind_farm.get_grid()
        img_name = f"farm_map_{grid['rows']}x{grid['cols']}.png"
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig.savefig(os.path.join(dir_name, img_name), dpi=300, bbox_inches='tight')
        print(f"Saved farm map to {os.path.join(dir_name, img_name)}")
    else:
        plt.show()
    
def plot_data(data, config, pause_interval=0.1, quiver_samples=35,
              show_streamwise=True, save_path=None, save_at_x=None, fps=10, dpi=150):
    """
    Main driver to visualize wake data. Coordinates setup, data prep, and animation loop.
    """
    plt.close('all')

    all_u = [np.asarray(d.U) for d in data]
    all_omg = [np.asarray(d.OmegaX) for d in data]

    levels = {
        'u': np.linspace(0, 1, 21) * np.nanmax(all_u) / config.Uhub,
        'omg': np.linspace(-1, 1, 21) * np.nanmax(np.abs(all_omg)) / (config.Uhub / config.D)
        }
    
    # 1. Prepare common data and grids
    grid_info = _get_grid_info(data[0], config, config.Zhub)
    wake_history = _extract_streamwise_history(data, config, grid_info) if show_streamwise else None
    
    # 2. Setup Figure and Axes
    fig, axes_dict = _setup_layout(show_streamwise and wake_history is not None)
    
    # 3. Animation / Plotting Loop
    # Container for mutable state (colorbars) to avoid 'nonlocal' mess
    plot_state = {'cbar_vort': None, 'cbar_vel': None, 'cbar_wake': None}

    def update_frame(entry):
        _render_snapshot(axes_dict, plot_state, entry, config, grid_info, quiver_samples, levels)
        if wake_history:
            _render_streamwise(axes_dict, plot_state, entry, config, grid_info, wake_history)

    # Pre-identify indices for snapshots to save
    save_indices = {}
    if save_at_x is not None:
        actual_x = np.array([d.X / config.D for d in data])
        for target in save_at_x:
            # Find the index of the data entry closest to the user's target x/D
            idx = np.argmin(np.abs(actual_x - target))
            save_indices[idx] = target

    # Execution
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        _save_animation(fig, data, update_frame, save_path, config.yaw, fps, dpi, pause_interval, save_indices)
    else:
        _show_live(data, update_frame, pause_interval)

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

def _render_snapshot(axes, state, entry, config, grid, q_samples, levels):
    """Handles the top row: Contour plots for current timestep."""
    # 1. Vorticity (Top Left)
    ax_vort = axes['vort']
    
    omg = np.asarray(entry.OmegaX)
    norm_factor = config.Uhub / config.D
    omg_norm = omg / norm_factor
    
    if 'cf_vort' not in state:
        ax_vort.set_aspect('equal', 'box')
        state['cf_vort'] = ax_vort.contourf(grid['yloc'], grid['zloc'], omg_norm,
                        levels=levels['omg'], cmap='RdBu_r', extend='both')
        state['title_vort'] = ax_vort.set_title(f'$\\Omega_x$ (Normalized) at X/D = {entry.X/config.D:.1f}')
        state['cbar_vort'] = ax_vort.figure.colorbar(state['cf_vort'], ax=ax_vort, label='Vorticity')
    else:
        state['cf_vort'].remove()
        state['cf_vort'] = ax_vort.contourf(grid['yloc'], grid['zloc'], omg_norm,
                                    levels=levels['omg'], cmap='RdBu_r', extend='both')
        state['title_vort'].set_text(f'$\\Omega_x$ (Normalized) at X/D = {entry.X/config.D:.1f}')
        
    # Quiver overlay
    state = _add_quiver(ax_vort, entry, state, grid, q_samples)

    # 2. Velocity (Top Right)
    ax_vel = axes['vel']

    u_def_norm = np.asarray(entry.U) / config.Uhub

    if 'cf_vel' not in state:
        ax_vel.set_aspect('equal', 'box')
        state['cf_vel'] = ax_vel.contourf(grid['yloc'], grid['zloc'], u_def_norm, levels=levels['u'], cmap='turbo')
        state['title_vel'] = ax_vel.set_title(f'U/Uhub at X/D = {entry.X/config.D:.1f}')
        state['cbar_vel'] = ax_vel.figure.colorbar(state['cf_vel'], ax=ax_vel, label='Normalized Streamwise Velocity')
    else:
        state['cf_vel'].remove()
        state['cf_vel'] = ax_vel.contourf(grid['yloc'], grid['zloc'], u_def_norm, levels=levels['u'], cmap='turbo')
        state['title_vel'].set_text(f'U/Uhub at X/D = {entry.X/config.D:.1f}')

def _add_quiver(ax, entry, state, grid, samples):
    """Adds quiver arrows to an existing axis."""
    dN = max(1, int(grid['Ny'] / samples))
    sl = np.s_[::dN, ::dN] # Slice object

    V_new = np.asarray(entry.V)[sl]
    W_new = np.asarray(entry.W)[sl]

    if 'quiver' not in state:
        state['quiver'] = ax.quiver(grid['yloc'][sl], grid['zloc'][sl], 
                V_new, W_new, color='k', scale=16.0, angles='xy', zorder=2)
    else:
        state['quiver'].set_UVC(V_new, W_new)
    return state

def _render_streamwise(axes, state, entry, config, grid, history):
    """Handles bottom row: Profiles and Heatmap."""
    current_x = entry.X / config.D
    
    # 3. Radial Profiles (Bottom Left)
    ax_prof = axes['prof']
    
    if 'prof_lines' not in state:
        # Plot background profiles (static snapshots)
        num_profs = min(5, len(history['X']))
        indices = np.linspace(0, len(history['X'])-1, num_profs, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, num_profs))
        
        u_norm = history['U'] / config._init_Uhub()
        for i, idx in enumerate(indices):
            ax_prof.plot(grid['yloc'][:,0], u_norm[:, idx], 
                    color=colors[i], marker='o', ms=3, alpha=0.5, label=f'X/D={history["X"][idx]:.1f}')
            
        # Highlight current profile
        curr_idx = np.argmin(np.abs(history['X'] - current_x))
        state['prof_lines'] = ax_prof.plot(grid['yloc'][:,0], u_norm[:, curr_idx], 'r-', lw=3, label='Current Profile')[0]
        
        ax_prof.set_title('Hub Height Velocity Profiles')
        ax_prof.set_ylabel('U/Uhub')
        ax_prof.set_xlabel('Normalized Cross-stream Distance Y/D')
        ax_prof.grid(True, alpha=0.3)
        ax_prof.legend(fontsize=8)
    else:
        curr_idx = np.argmin(np.abs(history['X'] - current_x))
        u_norm = history['U'] / config._init_Uhub()
        state['prof_lines'].set_ydata(u_norm[:, curr_idx])

    # 4. Wake Evolution Heatmap (Bottom Right)
    ax_wake = axes['wake']
    
    if 'v_line' not in state:
        u_norm = history['U'] / config._init_Uhub()
        mesh = ax_wake.pcolormesh(history['X_grid'], history['Y_grid'], u_norm, cmap='turbo', shading='auto', rasterized=True)
        state['v_line'] = ax_wake.axvline(current_x, color='r', ls='--')
        ax_wake.set_title('Wake Evolution (Top Down)')
        ax_wake.set_xlabel('Streamwise Distance X/D')
        ax_wake.set_ylabel('Normalized Cross-stream Distance Y/D')
        state['cbar_wake'] = ax_wake.figure.colorbar(mesh, ax=ax_wake, label='Normalized Velocity U/Uhub')
    else:
        state['v_line'].set_xdata([current_x, current_x])

def _save_animation(fig, data, update_func, path, yaw, fps, dpi, interval, save_indices):
    """Handles the video writing logic."""
    folder_name = f"Yaw{yaw:.2f}"
    os.makedirs(os.path.join(path, folder_name), exist_ok=True)
    try:
        out_path = os.path.join(path, folder_name, f"Yaw{yaw:.2f}_Wake.gif")
        writer = PillowWriter(fps=fps)
        with writer.saving(fig, out_path, dpi):
            for i, entry in enumerate(data):
                update_func(entry)
                fig.canvas.draw()

                if i in save_indices.keys():
                    fname = os.path.join(path, folder_name, f"Wake_xD_{save_indices[i]:.1f}.png")
                    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
                    print(f"Saved snapshot at x/D = {save_indices[i]:.1f}")

                writer.grab_frame()
                plt.pause(interval) # Optional: keep small pause to see progress
    except Exception as e:
        print(f"Error saving animation: {e}")

def _show_live(data, update_func, interval):
    """Handles standard live plotting."""
    for entry in data:
        update_func(entry)
        plt.pause(interval)

def npy_to_list(d):
    if isinstance(d, dict):
        return {k: npy_to_list(v) for k, v in d.items()}
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d