from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from simulation.data_structures import VortexField

def get_value(obj, key, default=None):
    """Extract value from either dict or object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def load_saved_results(out_path):
    """
    Load CSVs produced by WindFarm.save_results(out_path).

    Returns:
      dict keyed by turbine_index:
        {
          'yaw': float,
          'frames': [
            {'frame_index': int, 'X': float, 't': float,
             'yloc': (Ny,Nz) array, 'zloc': (Ny,Nz) array,
             'U': (Ny,Nz) array, 'V': ..., 'W': ..., 'OmegaX': ...},
            ...
          ],
          'csv': filename
        }
    """
    results = {}
    for fname in sorted(os.listdir(out_path)):
        if not fname.lower().endswith('.csv'):
            continue
        path = os.path.join(out_path, fname)
        df = pd.read_csv(path)
        if df.empty:
            continue
        ti = int(df['turbine_index'].iloc[0])
        yaw = float(df['yaw'].iloc[0])
        frames = []
        # group by frame_index and ensure ordering
        for fi, g in df.groupby('frame_index', sort=True):
            g_sorted = g.sort_values(['y_idx', 'z_idx'])
            Ny = int(g_sorted['y_idx'].max()) + 1
            Nz = int(g_sorted['z_idx'].max()) + 1
            # reshape in the same ordering as saved (y_idx major, z_idx minor)
            vals = lambda col: np.asarray(g_sorted[col].values).reshape(Ny, Nz)
            yloc = vals('y')
            zloc = vals('z')
            U = vals('U')
            V = vals('V')
            W = vals('W')
            Om = vals('OmegaX')
            X = float(g_sorted['X'].iloc[0])
            tval = float(g_sorted['t'].iloc[0])
            frames.append({
                'frame_index': int(fi), 'X': X, 't': tval,
                'yloc': yloc, 'zloc': zloc,
                'U': U, 'V': V, 'W': W, 'OmegaX': Om
            })
        results[ti] = {'yaw': yaw, 'frames': frames, 'csv': fname}
    return results

def turbine_frames_to_vortexfields(turbine_dict):
    """
    Optional: convert frames for a single turbine (one entry of load_saved_results)
    into VortexField-like objects if data_structures.VortexField is available.

    Returns list of VortexField instances (or dicts if import fails).
    """
    vfields = []
    for f in turbine_dict['frames']:
        if VortexField is not None:
            vf = VortexField(
                Y=None, Z=None, Rv=None, Circ=None,
                yloc=f['yloc'], zloc=f['zloc'],
                V=f.get('V'), W=f.get('W'),
                OmegaX=f.get('OmegaX'), t=f['t']
            )
            # set additional attributes used elsewhere
            vf.U = f.get('U')
            vf.X = f.get('X')
            vfields.append(vf)
        else:
            vfields.append(f)  # fallback: return raw dict
    return vfields

def plot_vorticity_contour(ax, state, entry, grid, q_samples, levels, D=1.0, Uhub=1.0):
    """Plot vorticity contour with quiver overlay.
    
    Args:
        entry: dict or object with OmegaX, X attributes/keys
        D: rotor diameter for normalization
        Uhub: hub velocity for normalization
    """
    omg = np.asarray(get_value(entry, 'OmegaX'))
    norm_factor = Uhub / D
    omg_norm = omg / norm_factor
    X_val = get_value(entry, 'X', 0.0)
    
    if 'cf_vort' not in state:
        ax.set_aspect('equal', 'box')
        state['cf_vort'] = ax.contourf(grid['yloc'], grid['zloc'], omg_norm,
                        levels=levels['omg'], cmap='RdBu_r', extend='both')
        state['title_vort'] = ax.set_title(f'$\\Omega_x$ (Normalized) at X/D = {X_val/D:.1f}')
        state['cbar_vort'] = ax.figure.colorbar(state['cf_vort'], ax=ax, label='Vorticity')
        state['cbar_vort'].ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        state['cf_vort'].remove()
        state['cf_vort'] = ax.contourf(grid['yloc'], grid['zloc'], omg_norm,
                                    levels=levels['omg'], cmap='RdBu_r', extend='both')
        state['title_vort'].set_text(f'$\\Omega_x$ (Normalized) at X/D = {X_val/D:.1f}')
        
    # Quiver overlay
    state = add_quiver(ax, entry, state, grid, q_samples)
    return state

def plot_velocity_contour(ax, state, entry, grid, levels, D=1.0, Uhub=1.0):
    """Plot velocity contour.
    
    Args:
        entry: dict or object with U, X attributes/keys
        D: rotor diameter for normalization
        Uhub: hub velocity for normalization
    """
    U_val = np.asarray(get_value(entry, 'U'))
    u_def_norm = U_val / Uhub
    X_val = get_value(entry, 'X', 0.0)

    if 'cf_vel' not in state:
        ax.set_aspect('equal', 'box')
        state['cf_vel'] = ax.contourf(grid['yloc'], grid['zloc'], u_def_norm, levels=levels['u'], cmap='turbo')
        state['title_vel'] = ax.set_title(f'U/Uhub at X/D = {X_val/D:.1f}')
        state['cbar_vel'] = ax.figure.colorbar(state['cf_vel'], ax=ax, label='Normalized Streamwise Velocity')
        state['cbar_vel'].ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        state['cf_vel'].remove()
        state['cf_vel'] = ax.contourf(grid['yloc'], grid['zloc'], u_def_norm, levels=levels['u'], cmap='turbo')
        state['title_vel'].set_text(f'U/Uhub at X/D = {X_val/D:.1f}')
    state = add_quiver(ax, entry, state, grid, samples=20)
    return state

def add_quiver(ax, entry, state, grid, samples):
    """Adds quiver arrows to an existing axis.
    
    Args:
        entry: dict or object with V, W attributes/keys
    """
    dN = max(1, int(grid['Ny'] / samples))
    sl = np.s_[::dN, ::dN] # Slice object

    V_new = np.asarray(get_value(entry, 'V'))[sl]
    W_new = np.asarray(get_value(entry, 'W'))[sl]

    if 'quiver' not in state:
        state['quiver'] = ax.quiver(grid['yloc'][sl], grid['zloc'][sl], 
                V_new, W_new, color='k', scale=16.0, angles='xy', zorder=2)
    else:
        state['quiver'].set_UVC(V_new, W_new)
    return state

def plot_radial_profiles(ax, state, entry, grid, history, D=1.0, Uhub=1.0):
    """Plot hub-height velocity profiles at various streamwise locations.
    
    Args:
        entry: dict or object with X attribute/key
        history: dict with 'X' and 'U' arrays
        D: rotor diameter for normalization
        Uhub: hub velocity for normalization
    """
    X_val = get_value(entry, 'X', 0.0)
    current_x = X_val / D
    
    if 'prof_lines' not in state:
        # Plot background profiles (static snapshots)
        num_profs = min(5, len(history['X']))
        indices = np.linspace(0, len(history['X'])-1, num_profs, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, num_profs))
        
        u_norm = history['U'] / Uhub
        for i, idx in enumerate(indices):
            ax.plot(grid['yloc'][:,0], u_norm[:, idx], 
                    color=colors[i], marker='o', ms=3, alpha=0.5, label=f'X/D={history["X"][idx]:.1f}')
            
        # Highlight current profile
        curr_idx = np.argmin(np.abs(history['X'] - current_x))
        state['prof_lines'] = ax.plot(grid['yloc'][:,0], u_norm[:, curr_idx], 'r-', lw=3, label='Current Profile')[0]
        
        ax.set_title('Hub Height Velocity Profiles')
        ax.set_ylabel('U/Uhub')
        ax.set_xlabel('Normalized Cross-stream Distance Y/D')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=16)
    else:
        curr_idx = np.argmin(np.abs(history['X'] - current_x))
        u_norm = history['U'] / Uhub
        state['prof_lines'].set_ydata(u_norm[:, curr_idx])
    return state

def plot_wake_evolution(ax, state, entry, history, D=1.0, Uhub=1.0):
    """Plot wake evolution heatmap with current position indicator.
    
    Args:
        entry: dict or object with X attribute/key
        history: dict with 'X_grid', 'Y_grid', and 'U' arrays
        D: rotor diameter for normalization
        Uhub: hub velocity for normalization
    """
    X_val = get_value(entry, 'X', 0.0)
    current_x = X_val / D
    
    if 'v_line' not in state:
        u_norm = history['U'] / Uhub
        mesh = ax.pcolormesh(history['X_grid'], history['Y_grid'], u_norm, cmap='turbo', shading='auto', rasterized=True)
        state['v_line'] = ax.axvline(current_x, color='r', ls='--')
        ax.set_title('Wake Evolution (Top Down)')
        ax.set_xlabel('Streamwise Distance X/D')
        ax.set_ylabel('Normalized Cross-stream Distance Y/D')
        state['cbar_wake'] = ax.figure.colorbar(mesh, ax=ax, label='Normalized Velocity U/Uhub')
    else:
        state['v_line'].set_xdata([current_x, current_x])
    return state

def plot_yaw_sweep_panel(data_path, turbine_idx, yaw_angles, x_positions, 
                         D=126.0, Uhub=8.0, plot_type='velocity', 
                         q_samples=20, figsize=None, cmap='turbo',
                         vmin=None, vmax=None, save_path=None):
    """
    Create a grid plot with yaw angles as rows and x/D positions as columns.
    
    Args:
        data_path: Path to folder containing CSV files (e.g., "Data/")
        turbine_idx: Turbine index to plot (e.g., 0)
        yaw_angles: List of yaw angles to plot (rows), e.g., [0.0, 10.0, 20.0]
        x_positions: List of x/D positions to plot (columns), e.g., [2, 4, 6, 8]
        D: Rotor diameter for normalization
        Uhub: Hub velocity for normalization
        plot_type: 'velocity' or 'vorticity'
        q_samples: Number of quiver samples (for vorticity plots)
        figsize: Figure size tuple (width, height). Auto-calculated if None
        cmap: Colormap ('turbo' for velocity, 'RdBu_r' for vorticity)
        vmin, vmax: Color scale limits (auto if None)
        save_path: Path to save figure (None to show instead)
    
    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    n_rows = len(yaw_angles)
    n_cols = len(x_positions)
    
    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (4 * n_cols, 3.5 * n_rows)
    
    # Load all data for specified yaw angles
    results_by_yaw = {}
    for yaw in yaw_angles:
        # Find CSV file for this yaw angle
        fname = f"Turbine_{turbine_idx}_Yaw{yaw:.2f}" + ".csv"
        fpath = os.path.join(data_path, fname)
        
        if not os.path.exists(fpath):
            print(f"Warning: File not found: {fpath}")
            continue
            
        df = pd.read_csv(fpath)
        if df.empty:
            continue
            
        # Group frames by frame_index
        frames = []
        for fi, g in df.groupby('frame_index', sort=True):
            g_sorted = g.sort_values(['y_idx', 'z_idx'])
            Ny = int(g_sorted['y_idx'].max()) + 1
            Nz = int(g_sorted['z_idx'].max()) + 1
            vals = lambda col: np.asarray(g_sorted[col].values).reshape(Ny, Nz)
            
            frames.append({
                'frame_index': int(fi), 'X': float(g_sorted['X'].iloc[0]), 
                't': float(g_sorted['t'].iloc[0]),
                'yloc': vals('y'), 'zloc': vals('z'),
                'U': vals('U'), 'V': vals('V'), 
                'W': vals('W'), 'OmegaX': vals('OmegaX')
            })
        results_by_yaw[yaw] = frames
    
    if not results_by_yaw:
        print("Error: No data loaded")
        return None, None
    
    # Setup grid from first available frame
    first_frames = next(iter(results_by_yaw.values()))
    frame0 = first_frames[0]
    grid = {
        'yloc': frame0['yloc'] / D,
        'zloc': frame0['zloc'] / D,
        'Ny': frame0['yloc'].shape[0],
        'Nz': frame0['yloc'].shape[1]
    }
    
    # Determine color scale limits if not provided
    if vmin is None or vmax is None:
        all_vals = []
        for frames in results_by_yaw.values():
            if plot_type == 'velocity':
                all_vals.extend([np.nanmax(f['U']) for f in frames])
            else:
                all_vals.extend([np.nanmax(np.abs(f['OmegaX'])) for f in frames])
        
        if plot_type == 'velocity':
            vmin = 0.0 if vmin is None else vmin
            vmax = (np.nanmax(all_vals) / Uhub) if vmax is None else vmax
        else:
            max_omg = np.nanmax(all_vals) / (Uhub / D)
            vmin = -max_omg if vmin is None else vmin
            vmax = max_omg if vmax is None else vmax
    
    # Define contour levels
    levels = np.linspace(vmin, vmax, 21)
    cmap = 'turbo' if plot_type == 'velocity' else 'RdBu_r'
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=False)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
    
    # Minimize spacing between subplots
    fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.04, hspace=-0.5, wspace=0.2)
    
    # Plot each panel
    for row_idx, yaw in enumerate(yaw_angles):
        if yaw not in results_by_yaw:
            continue
            
        frames = results_by_yaw[yaw]
        
        for col_idx, x_pos in enumerate(x_positions):
            ax = axes[row_idx, col_idx]
            
            # Find closest frame to desired x/D position
            X_values = np.array([f['X'] / D for f in frames])
            frame_idx = np.argmin(np.abs(X_values - x_pos))
            frame = frames[frame_idx]
            actual_x = frame['X'] / D
            
            # Plot based on type
            if plot_type == 'velocity':
                U_norm = frame['U'] / Uhub
                pcm = ax.contourf(grid['yloc'], grid['zloc'], U_norm, 
                                   cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
            else:
                omg_norm = frame['OmegaX'] / (Uhub / D)
                pcm = ax.contourf(grid['yloc'], grid['zloc'], omg_norm, 
                                   cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
                
                # Add quiver for vorticity plots
                dN = max(1, int(grid['Ny'] / q_samples))
                sl = np.s_[::dN, ::dN]
                V_sample = (frame['V'] / Uhub)[sl]
                W_sample = (frame['W'] / Uhub)[sl]
                ax.quiver(grid['yloc'][sl], grid['zloc'][sl], 
                            V_sample, W_sample, color='k', scale=1.0, 
                            angles='xy', alpha=0.6)
            
            # Formatting
            ax.set_aspect('equal', 'box')

            # Column titles (only top row)
            if row_idx == 0:
                ax.set_title(f'x/D = {actual_x:.1f}', fontsize=16)
            
            # Column titles (only top row)
            if row_idx == 0:
                ax.set_title(f'x/D = {actual_x:.1f}', fontsize=16)
            
            # Row labels (only left column)
            if col_idx == 0:
                ax.set_ylabel(f'γ = {yaw:.0f}°\nz/D', fontsize=16)
            else:
                ax.set_yticklabels([])
                
            # X-axis labels (only bottom row)
            if row_idx == n_rows - 1:
                ax.set_xlabel('y/D', fontsize=16)
            else:
                ax.set_xticklabels([])
    
    # Add single colorbar
    cbar_label = 'Normalized Streamwise Velocity U/Uhub' if plot_type == 'velocity' else r'Normalized Vorticity $\Omega_x D / U_{hub}$'
    cbar = fig.colorbar(pcm, ax=axes, orientation='vertical', 
                        fraction=0.02, pad=0.02, label=cbar_label)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_size(16)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    return fig, axes

def plot_wake_evolution_panel(data_path, turbine_idx, yaw_angles, 
                              D=126.0, Uhub=8.55, Z_hub=None, X_limit=None,
                              figsize=None, cmap='turbo',
                              vmin=None, vmax=None, draw_centerline=True, save_path=None):
    """
    Create a multi-row plot showing wake evolution for different yaw angles.
    Each row shows the streamwise (x/D) vs lateral (y/D) wake evolution at hub height.
    
    Args:
        data_path: Path to folder containing CSV files (e.g., "Data/")
        turbine_idx: Turbine index to plot (e.g., 0)
        yaw_angles: List of yaw angles to plot (one per row), e.g., [0.0, 15.0, 30.0]
        D: Rotor diameter for normalization
        Uhub: Hub velocity for normalization
        Z_hub: Hub height for extracting slice (None = middle of domain)
        X_limit: Maximum x/D to plot (None = all data)
        figsize: Figure size tuple (width, height). Auto-calculated if None
        cmap: Colormap (default 'turbo')
        vmin, vmax: Color scale limits (auto if None)
        draw_centerline: If True, draw wake centerline (minimum velocity trajectory)
        save_path: Path to save figure (None to show instead)
    
    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    n_rows = len(yaw_angles)
    
    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (14, 4 * n_rows)
    
    # Load all data for specified yaw angles
    results_by_yaw = {}
    for yaw in yaw_angles:
        # Find CSV file for this yaw angle
        fname = f"Turbine_{turbine_idx}_Yaw{yaw:.2f}" + ".csv"
        fpath = os.path.join(data_path, fname)
        
        if not os.path.exists(fpath):
            print(f"Warning: File not found: {fpath}")
            continue
            
        df = pd.read_csv(fpath)
        if df.empty:
            continue
            
        # Group frames by frame_index
        frames = []
        for fi, g in df.groupby('frame_index', sort=True):
            g_sorted = g.sort_values(['y_idx', 'z_idx'])
            Ny = int(g_sorted['y_idx'].max()) + 1
            Nz = int(g_sorted['z_idx'].max()) + 1
            vals = lambda col: np.asarray(g_sorted[col].values).reshape(Ny, Nz)
            
            if X_limit is None or (g_sorted['X'].iloc[0] / D) <= X_limit:
                frames.append({
                    'frame_index': int(fi), 'X': float(g_sorted['X'].iloc[0]), 
                    't': float(g_sorted['t'].iloc[0]),
                    'yloc': vals('y'), 'zloc': vals('z'),
                    'U': vals('U'), 'V': vals('V'), 
                    'W': vals('W'), 'OmegaX': vals('OmegaX')
                })
        results_by_yaw[yaw] = frames
    
    if not results_by_yaw:
        print("Error: No data loaded")
        return None, None
    
    # Setup grid from first available frame
    first_frames = next(iter(results_by_yaw.values()))
    frame0 = first_frames[0]
    Ny = frame0['yloc'].shape[0]
    Nz = frame0['yloc'].shape[1]
    
    # Determine hub height index if not provided
    if Z_hub is None:
        z_hub_idx = Nz // 2
    else:
        z_hub_idx = np.argmin(np.abs(frame0['zloc'][0, :] - Z_hub))
    
    # Extract streamwise evolution data for each yaw angle
    evolution_by_yaw = {}
    for yaw, frames in results_by_yaw.items():
        X_pos = []
        U_hub_profiles = []
        
        for frame in frames:
            X_pos.append(frame['X'] / D)
            # Extract hub-height slice (y vs x at fixed z)
            U_hub_profiles.append(frame['U'][:, z_hub_idx])
        
        # Create meshgrid for heatmap
        yloc_norm = frame0['yloc'][:, 0] / D
        X_grid, Y_grid = np.meshgrid(X_pos, yloc_norm)
        U_array = np.array(U_hub_profiles).T  # Shape: (Ny, N_frames)
        
        evolution_by_yaw[yaw] = {
            'X_grid': X_grid,
            'Y_grid': Y_grid,
            'U': U_array
        }
    
    # Determine color scale limits if not provided
    if vmin is None or vmax is None:
        all_U = [evo['U'] for evo in evolution_by_yaw.values()]
        if vmin is None:
            vmin = np.nanmin([np.nanmin(u) for u in all_U]) / Uhub
        if vmax is None:
            vmax = np.nanmax([np.nanmax(u) for u in all_U]) / Uhub
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, constrained_layout=False)
    if n_rows == 1:
        axes = [axes]
    
    # Plot each yaw angle
    for row_idx, yaw in enumerate(yaw_angles):
        if yaw not in evolution_by_yaw:
            continue
            
        ax = axes[row_idx]
        evo = evolution_by_yaw[yaw]
        
        # Normalize and plot
        U_norm = evo['U'] / Uhub
        levels = np.linspace(vmin, vmax, 21)
        mesh = ax.contourf(evo['X_grid'], evo['Y_grid'], U_norm, 
                            cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
        
        # Draw wake centerline (trajectory of minimum velocity)
        if draw_centerline:
            X_centers = evo['X_grid'][0, :]
            Y_centers = []
            for i in range(U_norm.shape[1]):
                # Find y position with minimum velocity at each x
                min_idx = np.nanargmin(U_norm[:, i])
                Y_centers.append(evo['Y_grid'][min_idx, i])
            Y_centers = np.array(Y_centers)
            
            # Plot centerline
            ax.plot(X_centers, Y_centers, 'w--', linewidth=2.5, label='Wake Center', zorder=5)
            ax.plot(X_centers, Y_centers, 'k--', linewidth=1.5, alpha=0.7, zorder=4)
            if row_idx == 0:
                ax.legend(loc='upper right', fontsize=12, framealpha=0.8)
        
        # Formatting
        ax.set_ylabel(f'γ = {yaw:.0f}°\ny/D', fontsize=16)
        # ax.set_title(f'Wake Evolution at γ = {yaw:.0f}°', fontsize=16, pad=10)
        
        # X-axis labels (only bottom row)
        if row_idx == n_rows - 1:
            ax.set_xlabel('Streamwise Distance x/D', fontsize=16)
        else:
            ax.set_xticklabels([])
    
    # Add single colorbar
    cbar = fig.colorbar(mesh, ax=axes, orientation='vertical', 
                       fraction=0.02, pad=0.02, label='Normalized Velocity U/Uhub')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_size(16)
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    return fig, axes

if __name__ == "__main__":
    os.makedirs("Figures", exist_ok=True)
    # Yaw sweep panel (cross-sections at different x/D)
    plot_yaw_sweep_panel(
        data_path="Data/",
        turbine_idx=0,
        yaw_angles=[0.0, 15.0, 30.0],
        x_positions=[2, 4, 6, 8],
        D=126.0,
        Uhub=8.55,
        plot_type='vorticity',  # or 'vorticity'
        save_path='Figures/yaw_sweep_vorticity.png'
    )

    plot_yaw_sweep_panel(
        data_path="Data/",
        turbine_idx=0,
        yaw_angles=[0.0, 15.0, 30.0],
        x_positions=[2, 4, 6, 8],
        D=126.0,
        Uhub=8.55,
        plot_type='velocity',  # or 'vorticity'
        save_path='Figures/yaw_sweep_velocity.png'
    )
    
    # Wake evolution panel (streamwise evolution)
    plot_wake_evolution_panel(
        data_path="Data/",
        turbine_idx=0,
        yaw_angles=[0.0, 15.0, 30.0],
        X_limit=10.0,
        Z_hub=90.0,
        D=126.0,
        Uhub=8.55,
        vmin=0.4,
        save_path='Figures/wake_evolution_yaw.png'
    )