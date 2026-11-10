from scipy.ndimage import uniform_filter, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patheffects as patheffects
import numpy as np
import jax.numpy as jnp
import os

from .field_ops import interpolate_local_velocity_field
from .viz_style import SURFACE, INK_PRIMARY, INK_SECONDARY, style_figure, style_axes, style_colorbar

DU_KV1, DU_KV0, DU_XV1 = 0.05, 0.001, 0.5
IU_TO_TI = 1.28


def du_nu_t(TI, age_D, U0, D):
    """Du Eq (26): nu_t/(U0 D) = (kv1 TI - kv0) * min(age/D, xv1/TI)."""
    return (jnp.maximum(DU_KV1 * TI - DU_KV0, 0.0)
            * jnp.minimum(age_D, DU_XV1 / TI) * U0 * D)


def ambient_tke(I_amb, U0):
    """k_B from TI = sqrt(2k/3)/U0, with TI the ambient value Iu/1.28."""
    return 1.5 * ((I_amb / IU_TO_TI) * U0) ** 2


def conservation_integrals(turbine):
    """Per-bay conservation of the two transported integrals, as ratios to their 1 D value."""
    live = [f for f in turbine.wake_field if np.size(f.U)]
    if len(live) < 3:
        return None
    y = np.asarray(turbine.yloc)[:, 0]
    z = np.asarray(turbine.zloc)[0, :]
    dA = (y[1] - y[0]) * (z[1] - z[0])
    Uin = np.asarray(turbine.Uin)
    xs, Ms, Ks = [], [], []
    for f in live:
        U = np.asarray(f.U)
        xs.append(float(f.X) / turbine.D)
        Ms.append(float(np.sum(np.maximum(Uin - U, 0.0) * U) * dA))
        kw = np.asarray(f.kw) if np.size(getattr(f, "kw", [])) else None
        Ks.append(float(np.sum(kw * U) * dA) if kw is not None else np.nan)
    j = next((k for k, x in enumerate(xs) if x > 1.0), 0)
    return {"x_end": xs[-1],
            "M_ratio": Ms[-1] / Ms[j] if Ms[j] else float("nan"),
            "K_ratio": Ks[-1] / Ks[j] if Ks[j] else float("nan")}


def report_conservation(wind_farm, first_bay=3):
    """One line per farm: conservation_integrals for every bay from `first_bay` on."""
    parts = []
    for i, t in enumerate(wind_farm.turbines):
        if i + 1 < first_bay:
            continue
        c = conservation_integrals(t)
        if c is None:
            continue
        parts.append("T%d M%+.1f%% K%+.1f%%"
                     % (i + 1, 100 * (c["M_ratio"] - 1), 100 * (c["K_ratio"] - 1)))
    if parts:
        print("  conservation, bays %d+ (M = Int Uw U dA, K = Int k_w U dA, ratio to 1 D)"
              % first_bay)
        print("    " + "  ".join(parts))
    return parts


def div_nu_grad(nu, f, dy, dz):
    """div(nu grad f) on the three-point stencil, in conservative flux form."""
    qy = 0.5 * (nu[:-1, :] + nu[1:, :]) * (f[1:, :] - f[:-1, :]) / dy
    out = jnp.zeros_like(f).at[1:-1, :].add((qy[1:, :] - qy[:-1, :]) / dy)
    qz = 0.5 * (nu[:, :-1] + nu[:, 1:]) * (f[:, 1:] - f[:, :-1]) / dz
    return out.at[:, 1:-1].add((qz[:, 1:] - qz[:, :-1]) / dz)


def velocity_floor(Uin):
    """Lower bound on U wherever it appears in a denominator."""
    return jnp.maximum(0.1 * Uin, 1e-6)


def du_eps_coef(TI, D):
    """Du's dissipation coefficient, from his C_eps and l_m: eps = C_eps k_w^(3/2) / l_m."""
    return (0.4 * TI + 0.010) / ((0.2 * TI + 0.015) * D)


def advance_kw(kw, U, Uin, yloc, zloc, dx, nu_t, eps_coef, V, W):
    """One march step of Du et al. (2025) Eq (13), the wake-added TKE budget, with the shear production
    taken against the FULL mean gradient, nu_t dDu/dx_j dU/dx_j (Klemmer & Howland 2025, Eq 8; the
    dominant term of Du's own LES budget, Eq 9, before his azimuthal averaging): the inflow shear adds
    TKE above the hub and removes some below it."""
    Uw = Uin - U
    dWdY, dWdZ = jnp.gradient(Uw, yloc[:, 0], zloc[0, :])
    dUindZ = jnp.gradient(Uin, zloc[0, :], axis=1)
    prod = nu_t * (dWdY ** 2 + dWdZ ** 2 - dWdZ * dUindZ)
    dif = div_nu_grad(nu_t, kw, yloc[1, 0] - yloc[0, 0], zloc[0, 1] - zloc[0, 0])
    dkdY, dkdZ = jnp.gradient(kw, yloc[:, 0], zloc[0, :])
    rhs = dif - eps_coef * jnp.maximum(kw, 0.0) ** 1.5 + prod - V * dkdY - W * dkdZ

    kw = jnp.maximum(kw + (dx / jnp.maximum(U, velocity_floor(Uin))) * rhs, 0.0)

    kw = kw.at[:, 0].set(kw[:, 1])
    kw = kw.at[0, :].set(kw[1, :])
    return kw.at[-1, :].set(kw[-2, :])


def advect_incident_ti(ti, U, Uin, yloc, zloc, dx, V, W, nu_t=None):
    """Carry the incident-TI field with the flow:"""
    dtiY, dtiZ = jnp.gradient(ti, yloc[:, 0], zloc[0, :])
    rhs = -V * dtiY - W * dtiZ
    if nu_t is not None:
        rhs = rhs + div_nu_grad(nu_t, ti, yloc[1, 0] - yloc[0, 0], zloc[0, 1] - zloc[0, 0])
    ti = ti + (dx / jnp.maximum(U, velocity_floor(Uin))) * rhs
    ti = ti.at[:, 0].set(ti[:, 1])
    ti = ti.at[0, :].set(ti[1, :])
    return jnp.maximum(ti.at[-1, :].set(ti[-2, :]), 0.0)


def smooth_2d(U, kernel_size=3, method='gaussian'):
    """2D smoothing with options: - method='gaussian' : Gaussian blur (smooth, continuous look)."""
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

    print("Generating Farm Deficit Map...")

    elevation_func = wind_farm.turbine_configs.elevation_func

    all_x = [t.pos[0] for t in wind_farm.turbines]
    all_y = [t.pos[1] for t in wind_farm.turbines]
    D = wind_farm.turbines[0].D
    Uhub_ref = wind_farm.turbines[0].Uhub
    Zhub_ref = wind_farm.turbines[0].Zhub + wind_farm.turbines[0].pos[2]
    y_slice_val = wind_farm.turbines[0].pos[1]
    z_min_plot = 0.0

    min_wake_len = wind_farm.field_params.min_X * D
    max_wake_wid = wind_farm.field_params.max_Y * D
    max_wake_hgt = wind_farm.field_params.max_Z * D

    x_min = min(all_x) - 2 * D
    x_max = max(all_x) + min_wake_len
    y_min = min(all_y) - max_wake_wid
    y_max = max(all_y) + max_wake_wid
    z_min = 0.0
    z_max = max([(t.Zhub + t.pos[2]) for t in wind_farm.turbines]) + max_wake_hgt

    X_vis = np.linspace(x_min, x_max, x_resolution)
    Y_vis = np.linspace(y_min, y_max, y_resolution)
    Z_vis = np.linspace(z_min, z_max, z_resolution)

    Y_loc, Z_loc = np.meshgrid(Y_vis, Z_vis, indexing='ij')

    fp = wind_farm.field_params
    z_safe = np.maximum(Z_loc, fp.z0 + 1e-3)
    U_in = fp.Uh * (np.log(z_safe / fp.z0) / np.log(fp.Zh / fp.z0))

    z_ref_idx = np.argmin(np.abs(Z_vis - Zhub_ref))
    y_ref_idx = np.argmin(np.abs(Y_vis - y_slice_val))

    U_xy_map = np.zeros((len(Y_vis), len(X_vis)))
    U_xz_map = np.zeros((len(Z_vis), len(X_vis)))

    for i, x_global in enumerate(X_vis):
        src = max((t for t in wind_farm.turbines if t.pos[0] < x_global),
                  key=lambda t: t.pos[0], default=None)
        if src is None:
            U_total_slice = U_in
        else:
            u_abs, _ = interpolate_local_velocity_field(
                src, x_global - src.pos[0], Y_loc, Z_loc, default=U_in)
            U_total_slice = np.asarray(u_abs)

        U_xy_map[:, i] = U_total_slice[:, z_ref_idx]

        U_xz_map[:, i] = U_total_slice[y_ref_idx, :]

    U_xy_map = smooth_2d(U_xy_map, method='gaussian')
    U_xz_map = smooth_2d(U_xz_map, method='gaussian')

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))

    X_grid_xy, Y_grid_xy = np.meshgrid(X_vis, Y_vis)
    vmin, vmax = 0.2, 1.2
    levels = np.linspace(vmin, vmax, 50)

    im1 = ax1.contourf(X_grid_xy, Y_grid_xy, U_xy_map / Uhub_ref,
                       cmap='bwr', levels=levels, vmin=vmin, vmax=vmax, extend='both')

    X_grid_xz, Z_grid_xz = np.meshgrid(X_vis, Z_vis)

    im2 = ax2.contourf(X_grid_xz, Z_grid_xz, U_xz_map / Uhub_ref,
                       cmap='bwr', levels=levels, vmin=vmin, vmax=vmax, extend='both')

    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label(r'Normalized Velocity $U / U_{hub}$')
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.set_xlabel('Global Streamwise X (m)')
    ax1.set_ylabel('Global Cross-stream Y (m)')
    ax1.set_title(f'Top View (Z = {Zhub_ref:.1f}m)')

    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label(r'Normalized Velocity $U / U_{hub}$')
    cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_xlabel('Global Streamwise X (m)')
    ax2.set_ylabel('Global Vertical Z (m)')
    ax2.set_title('Side View (Y = 0m)')

    for idx, t in enumerate(wind_farm.turbines):
        yaw_rad = np.deg2rad(t.yaw)

        dx = (t.D / 2) * np.sin(yaw_rad)
        dy = (t.D / 2) * np.cos(yaw_rad)

        ax1.plot([t.pos[0] - dx, t.pos[0] + dx],
                 [t.pos[1] + dy, t.pos[1] - dy],
                 color='black', lw=3, solid_capstyle='round')

        ax1.text(t.pos[0], t.pos[1] + t.D * 0.6, f"T{idx}", color='white',
                 ha='center', va='center', fontweight='bold',
                 path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])

        ax2.plot([t.pos[0], t.pos[0]],
                 [t.pos[2], t.Zhub+t.pos[2]],
                 color='black', lw=2)

        z_top = t.pos[2] + t.Zhub + (t.D / 2)
        z_bot = t.pos[2] + t.Zhub - (t.D / 2)

        ax2.plot([t.pos[0], t.pos[0]],
                 [z_bot, z_top],
                 color='black', lw=4, solid_capstyle='round')

    if elevation_func:
        z_ground_line = elevation_func(X_vis, np.full_like(X_vis, y_slice_val))
        ax2.fill_between(X_vis, z_min_plot, z_ground_line, color='#6d4c41', alpha=1.0, zorder=5)
        ax2.plot(X_vis, z_ground_line, color='black', lw=1, zorder=6)

    plt.tight_layout()

    if save_path:
        dir_name = save_path
        grid = wind_farm.get_grid()
        img_name = f"farm_map_{grid['rows']}x{grid['cols']}.png"
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig.savefig(os.path.join(dir_name, img_name), dpi=300, bbox_inches='tight')
        print(f"Saved farm map to {os.path.join(dir_name, img_name)}")
    else:
        plt.show()

def plot_turbine_layout(wind_farm, save_path=None):
    """Top-down farm layout: rotors colored by yaw (diverging, since yaw is signed), labeled with..."""
    turbines = wind_farm.turbines
    if not turbines:
        print("No turbines found in wind farm.")
        return

    positions = np.array([t.pos[:2] for t in turbines])
    yaws = np.array([float(t.yaw) for t in turbines])
    D = np.array([float(t.D) for t in turbines])
    eff = np.array([float(t.calculate_efficiency()) for t in turbines])

    yaw_abs_max = max(1.0, np.max(np.abs(yaws)))
    norm = TwoSlopeNorm(vmin=-yaw_abs_max, vcenter=0.0, vmax=yaw_abs_max)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 5))
    style_figure(fig)
    ax.set_aspect('equal')

    for (x, y), yaw, d, e in zip(positions, yaws, D, eff):
        color = plt.cm.RdBu_r(norm(yaw))
        circ = plt.Circle((x, y), d / 2, facecolor=color, edgecolor=INK_SECONDARY, linewidth=1.2, zorder=3)
        ax.add_patch(circ)
        ax.annotate(f"η={e:.2f}\nyaw {yaw:+.0f}°", (x, y + d / 2), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, color=INK_PRIMARY)

    pad = D.max() * 2
    ax.set_xlim(positions[:, 0].min() - pad, positions[:, 0].max() + pad)
    ax.set_ylim(positions[:, 1].min() - pad, positions[:, 1].max() + pad * 1.4)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Wind farm layout — yaw & efficiency by turbine', loc='left')
    style_axes(ax)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r')
    cbar = fig.colorbar(mappable, ax=ax, label='Yaw angle (deg)')
    style_colorbar(cbar)

    fig.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, 'turbine_layout.png')
        fig.savefig(fname, dpi=300, bbox_inches='tight', facecolor=SURFACE)
        print(f"Saved turbine layout to {fname}")
    else:
        plt.show()

def plot_data(data, config, pause_interval=0.1, quiver_samples=35,
              show_streamwise=True, save_path=None, save_at_x=None, fps=10, dpi=150, show=True):
    """Main driver to visualize wake data."""
    if not show and save_path:
        import matplotlib
        matplotlib.use('Agg')

    plt.close('all')

    all_u = [np.asarray(d.U) for d in data]
    all_omg = [np.asarray(d.OmegaX) for d in data]

    levels = {
        'u': np.linspace(0, 1, 21) * ceil(np.nanmax(all_u) / config.Uhub, 1),
        'omg': np.linspace(-1, 1, 21) * ceil(np.nanmax(np.abs(all_omg)) / (config.Uhub / config.D), 1)
        }

    grid_info = _get_grid_info(data[0], config, config.Zhub)
    wake_history = _extract_streamwise_history(data, config, grid_info) if show_streamwise else None

    fig, axes_dict = _setup_layout(show_streamwise and wake_history is not None)

    plot_state = {'cbar_vort': None, 'cbar_vel': None, 'cbar_wake': None}

    def update_frame(entry):
        _render_snapshot(axes_dict, plot_state, entry, config, grid_info, quiver_samples, levels)
        if wake_history:
            _render_streamwise(axes_dict, plot_state, entry, config, grid_info, wake_history)

    save_indices = {}
    if save_at_x is not None:
        actual_x = np.array([d.X / config.D for d in data])
        for target in save_at_x:
            idx = np.argmin(np.abs(actual_x - target))
            save_indices[idx] = target

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        _save_animation(fig, data, update_frame, save_path, config.yaw, fps, dpi, pause_interval if show else 0, save_indices)
    elif show:
        _show_live(data, update_frame, pause_interval)

    return fig, axes_dict

def _get_grid_info(data_snapshot, config, z_target):
    """Normalizes grids and finds indices for hub-height slicing."""
    yloc = np.asarray(data_snapshot.yloc) / config.D
    zloc = np.asarray(data_snapshot.zloc) / config.D

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
        U_hub_profiles.append(np.asarray(d.U)[:, grid['z_hub_idx']])

    if len(X_pos) < 2: return None

    X_grid, Y_grid = np.meshgrid(X_pos, grid['yloc'][:, 0])

    return {
        'X': np.array(X_pos),
        'U': np.array(U_hub_profiles).T,
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
        state['cbar_vort'].ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        state['cf_vort'].remove()
        state['cf_vort'] = ax_vort.contourf(grid['yloc'], grid['zloc'], omg_norm,
                                    levels=levels['omg'], cmap='RdBu_r', extend='both')
        state['title_vort'].set_text(f'$\\Omega_x$ (Normalized) at X/D = {entry.X/config.D:.1f}')

    state = _add_quiver(ax_vort, entry, state, grid, q_samples)

    ax_vel = axes['vel']

    u_def_norm = np.asarray(entry.U) / config.Uhub

    if 'cf_vel' not in state:
        ax_vel.set_aspect('equal', 'box')
        state['cf_vel'] = ax_vel.contourf(grid['yloc'], grid['zloc'], u_def_norm, levels=levels['u'], cmap='turbo')
        state['title_vel'] = ax_vel.set_title(f'U/Uhub at X/D = {entry.X/config.D:.1f}')
        state['cbar_vel'] = ax_vel.figure.colorbar(state['cf_vel'], ax=ax_vel, label='Normalized Streamwise Velocity')
        state['cbar_vel'].ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        state['cf_vel'].remove()
        state['cf_vel'] = ax_vel.contourf(grid['yloc'], grid['zloc'], u_def_norm, levels=levels['u'], cmap='turbo')
        state['title_vel'].set_text(f'U/Uhub at X/D = {entry.X/config.D:.1f}')

def _add_quiver(ax, entry, state, grid, samples):
    """Adds quiver arrows to an existing axis."""
    dN = max(1, int(grid['Ny'] / samples))
    sl = np.s_[::dN, ::dN]

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

    ax_prof = axes['prof']

    if 'prof_lines' not in state:
        num_profs = min(5, len(history['X']))
        indices = np.linspace(0, len(history['X'])-1, num_profs, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, num_profs))

        u_norm = history['U'] / config.Uinf
        for i, idx in enumerate(indices):
            ax_prof.plot(grid['yloc'][:,0], u_norm[:, idx],
                    color=colors[i], marker='o', ms=3, alpha=0.5, label=f'X/D={history["X"][idx]:.1f}')

        curr_idx = np.argmin(np.abs(history['X'] - current_x))
        state['prof_lines'] = ax_prof.plot(grid['yloc'][:,0], u_norm[:, curr_idx], 'r-', lw=3, label='Current Profile')[0]

        ax_prof.set_title('Hub Height Velocity Profiles')
        ax_prof.set_ylabel('U/Uhub')
        ax_prof.set_xlabel('Normalized Cross-stream Distance Y/D')
        ax_prof.grid(True, alpha=0.3)
        ax_prof.legend(fontsize=8)
    else:
        curr_idx = np.argmin(np.abs(history['X'] - current_x))
        u_norm = history['U'] / config.Uinf
        state['prof_lines'].set_ydata(u_norm[:, curr_idx])

    ax_wake = axes['wake']

    if 'v_line' not in state:
        u_norm = history['U'] / config.Uinf
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
                if interval > 0:
                    plt.pause(interval)
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

def ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)
