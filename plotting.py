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
    """Load CSVs produced by WindFarm.save_results(out_path)."""
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
        for fi, g in df.groupby('frame_index', sort=True):
            g_sorted = g.sort_values(['y_idx', 'z_idx'])
            Ny = int(g_sorted['y_idx'].max()) + 1
            Nz = int(g_sorted['z_idx'].max()) + 1
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
    """Optional: convert frames for a single turbine (one entry of load_saved_results) into..."""
    vfields = []
    for f in turbine_dict['frames']:
        if VortexField is not None:
            vf = VortexField(
                Y=None, Z=None, Rv=None, Circ=None,
                yloc=f['yloc'], zloc=f['zloc'],
                V=f.get('V'), W=f.get('W'),
                OmegaX=f.get('OmegaX'), t=f['t']
            )
            vf.U = f.get('U')
            vf.X = f.get('X')
            vfields.append(vf)
        else:
            vfields.append(f)
    return vfields

def plot_vorticity_contour(ax, state, entry, grid, q_samples, levels, D=1.0, Uhub=1.0):
    """Plot vorticity contour with quiver overlay."""
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

    state = add_quiver(ax, entry, state, grid, q_samples)
    return state

def plot_velocity_contour(ax, state, entry, grid, levels, D=1.0, Uhub=1.0):
    """Plot velocity contour."""
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
    """Adds quiver arrows to an existing axis."""
    dN = max(1, int(grid['Ny'] / samples))
    sl = np.s_[::dN, ::dN]

    V_new = np.asarray(get_value(entry, 'V'))[sl]
    W_new = np.asarray(get_value(entry, 'W'))[sl]

    if 'quiver' not in state:
        state['quiver'] = ax.quiver(grid['yloc'][sl], grid['zloc'][sl],
                V_new, W_new, color='k', scale=16.0, angles='xy', zorder=2)
    else:
        state['quiver'].set_UVC(V_new, W_new)
    return state

def plot_radial_profiles(ax, state, entry, grid, history, D=1.0, Uhub=1.0):
    """Plot hub-height velocity profiles at various streamwise locations."""
    X_val = get_value(entry, 'X', 0.0)
    current_x = X_val / D

    if 'prof_lines' not in state:
        num_profs = min(5, len(history['X']))
        indices = np.linspace(0, len(history['X'])-1, num_profs, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, num_profs))

        u_norm = history['U'] / Uhub
        for i, idx in enumerate(indices):
            ax.plot(grid['yloc'][:,0], u_norm[:, idx],
                    color=colors[i], marker='o', ms=3, alpha=0.5, label=f'X/D={history["X"][idx]:.1f}')

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
    """Plot wake evolution heatmap with current position indicator."""
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
    """Create a grid plot with yaw angles as rows and x/D positions as columns."""
    n_rows = len(yaw_angles)
    n_cols = len(x_positions)

    if figsize is None:
        figsize = (4 * n_cols, 3.5 * n_rows)

    results_by_yaw = {}
    for yaw in yaw_angles:
        fname = f"Turbine_{turbine_idx}_Yaw{yaw:.2f}" + ".csv"
        fpath = os.path.join(data_path, fname)

        if not os.path.exists(fpath):
            print(f"Warning: File not found: {fpath}")
            continue

        df = pd.read_csv(fpath)
        if df.empty:
            continue

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

    first_frames = next(iter(results_by_yaw.values()))
    frame0 = first_frames[0]
    grid = {
        'yloc': frame0['yloc'] / D,
        'zloc': frame0['zloc'] / D,
        'Ny': frame0['yloc'].shape[0],
        'Nz': frame0['yloc'].shape[1]
    }

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

    levels = np.linspace(vmin, vmax, 21)
    cmap = 'turbo' if plot_type == 'velocity' else 'RdBu_r'

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=False)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.04, hspace=-0.5, wspace=0.2)

    for row_idx, yaw in enumerate(yaw_angles):
        if yaw not in results_by_yaw:
            continue

        frames = results_by_yaw[yaw]

        for col_idx, x_pos in enumerate(x_positions):
            ax = axes[row_idx, col_idx]

            X_values = np.array([f['X'] / D for f in frames])
            frame_idx = np.argmin(np.abs(X_values - x_pos))
            frame = frames[frame_idx]
            actual_x = frame['X'] / D

            if plot_type == 'velocity':
                U_norm = frame['U'] / Uhub
                pcm = ax.contourf(grid['yloc'], grid['zloc'], U_norm,
                                   cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
            else:
                omg_norm = frame['OmegaX'] / (Uhub / D)
                pcm = ax.contourf(grid['yloc'], grid['zloc'], omg_norm,
                                   cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)

                dN = max(1, int(grid['Ny'] / q_samples))
                sl = np.s_[::dN, ::dN]
                V_sample = (frame['V'] / Uhub)[sl]
                W_sample = (frame['W'] / Uhub)[sl]
                ax.quiver(grid['yloc'][sl], grid['zloc'][sl],
                            V_sample, W_sample, color='k', scale=1.0,
                            angles='xy', alpha=0.6)

            ax.set_aspect('equal', 'box')

            if row_idx == 0:
                ax.set_title(f'x/D = {actual_x:.1f}', fontsize=16)

            if row_idx == 0:
                ax.set_title(f'x/D = {actual_x:.1f}', fontsize=16)

            if col_idx == 0:
                ax.set_ylabel(f'γ = {yaw:.0f}°\nz/D', fontsize=16)
            else:
                ax.set_yticklabels([])

            if row_idx == n_rows - 1:
                ax.set_xlabel('y/D', fontsize=16)
            else:
                ax.set_xticklabels([])

    cbar_label = 'Normalized Streamwise Velocity U/Uhub' if plot_type == 'velocity' else r'Normalized Vorticity $\Omega_x D / U_{hub}$'
    cbar = fig.colorbar(pcm, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.02, label=cbar_label)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_size(16)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

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
    """Create a multi-row plot showing wake evolution for different yaw angles."""
    n_rows = len(yaw_angles)

    if figsize is None:
        figsize = (14, 4 * n_rows)

    results_by_yaw = {}
    for yaw in yaw_angles:
        fname = f"Turbine_{turbine_idx}_Yaw{yaw:.2f}" + ".csv"
        fpath = os.path.join(data_path, fname)

        if not os.path.exists(fpath):
            print(f"Warning: File not found: {fpath}")
            continue

        df = pd.read_csv(fpath)
        if df.empty:
            continue

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

    first_frames = next(iter(results_by_yaw.values()))
    frame0 = first_frames[0]
    Ny = frame0['yloc'].shape[0]
    Nz = frame0['yloc'].shape[1]

    if Z_hub is None:
        z_hub_idx = Nz // 2
    else:
        z_hub_idx = np.argmin(np.abs(frame0['zloc'][0, :] - Z_hub))

    evolution_by_yaw = {}
    for yaw, frames in results_by_yaw.items():
        X_pos = []
        U_hub_profiles = []

        for frame in frames:
            X_pos.append(frame['X'] / D)
            U_hub_profiles.append(frame['U'][:, z_hub_idx])

        yloc_norm = frame0['yloc'][:, 0] / D
        X_grid, Y_grid = np.meshgrid(X_pos, yloc_norm)
        U_array = np.array(U_hub_profiles).T

        evolution_by_yaw[yaw] = {
            'X_grid': X_grid,
            'Y_grid': Y_grid,
            'U': U_array
        }

    if vmin is None or vmax is None:
        all_U = [evo['U'] for evo in evolution_by_yaw.values()]
        if vmin is None:
            vmin = np.nanmin([np.nanmin(u) for u in all_U]) / Uhub
        if vmax is None:
            vmax = np.nanmax([np.nanmax(u) for u in all_U]) / Uhub

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, constrained_layout=False)
    if n_rows == 1:
        axes = [axes]

    for row_idx, yaw in enumerate(yaw_angles):
        if yaw not in evolution_by_yaw:
            continue

        ax = axes[row_idx]
        evo = evolution_by_yaw[yaw]

        U_norm = evo['U'] / Uhub
        levels = np.linspace(vmin, vmax, 21)
        mesh = ax.contourf(evo['X_grid'], evo['Y_grid'], U_norm,
                            cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)

        if draw_centerline:
            X_centers = evo['X_grid'][0, :]
            Y_centers = []
            for i in range(U_norm.shape[1]):
                min_idx = np.nanargmin(U_norm[:, i])
                Y_centers.append(evo['Y_grid'][min_idx, i])
            Y_centers = np.array(Y_centers)

            ax.plot(X_centers, Y_centers, 'w--', linewidth=2.5, label='Wake Center', zorder=5)
            ax.plot(X_centers, Y_centers, 'k--', linewidth=1.5, alpha=0.7, zorder=4)
            if row_idx == 0:
                ax.legend(loc='upper right', fontsize=12, framealpha=0.8)

        ax.set_ylabel(f'γ = {yaw:.0f}°\ny/D', fontsize=16)

        if row_idx == n_rows - 1:
            ax.set_xlabel('Streamwise Distance x/D', fontsize=16)
        else:
            ax.set_xticklabels([])

    cbar = fig.colorbar(mesh, ax=axes, orientation='vertical',
                       fraction=0.02, pad=0.02, label='Normalized Velocity U/Uhub')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_size(16)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    return fig, axes

if __name__ == "__main__":
    os.makedirs("Figures", exist_ok=True)
    plot_yaw_sweep_panel(
        data_path="runs/",
        turbine_idx=0,
        yaw_angles=[0.0, 15.0, 30.0],
        x_positions=[2, 4, 6, 8],
        D=126.0,
        Uhub=8.55,
        plot_type='vorticity',
        save_path='Figures/yaw_sweep_vorticity.png'
    )

    plot_yaw_sweep_panel(
        data_path="runs/",
        turbine_idx=0,
        yaw_angles=[0.0, 15.0, 30.0],
        x_positions=[2, 4, 6, 8],
        D=126.0,
        Uhub=8.55,
        plot_type='velocity',
        save_path='Figures/yaw_sweep_velocity.png'
    )

    plot_wake_evolution_panel(
        data_path="runs/",
        turbine_idx=0,
        yaw_angles=[0.0, 15.0, 30.0],
        X_limit=10.0,
        Z_hub=90.0,
        D=126.0,
        Uhub=8.55,
        vmin=0.4,
        save_path='Figures/wake_evolution_yaw.png'
    )
