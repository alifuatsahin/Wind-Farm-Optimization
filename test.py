def plot_farm_deficit_map(wind_farm, x_resolution=200, y_resolution=100, save_path=None):
    """
    Calculates the combined superposed velocity deficit on an X-Y grid at hub height.
    Uses a sum-of-squares superposition model.
    """
    
    if not wind_farm.turbines:
        print("No turbines found in wind farm.")
        return None, None, None

    # 1. Define Visualization Domain (Global Coordinates)
    # Gather extents based on all turbines
    all_x = [t.pos[0] for t in wind_farm.turbines]
    all_y = [t.pos[1] for t in wind_farm.turbines]
    
    # Use field params if available, otherwise guess reasonable bounds
    max_wake_len = wind_farm.field_params.max_X * wind_farm.turbines[0].D
    max_wake_wid = wind_farm.field_params.max_Y * wind_farm.turbines[0].D
    
    x_min = min(all_x) - 2 * wind_farm.turbines[0].D
    x_max = max(all_x) + max_wake_len
    y_min = min(all_y) - max_wake_wid
    y_max = max(all_y) + max_wake_wid

    X_vis = np.linspace(x_min, x_max, x_resolution)
    Y_vis = np.linspace(y_min, y_max, y_resolution)
    X_grid, Y_grid = np.meshgrid(X_vis, Y_vis, indexing='xy') # Shape (Ny, Nx)
    
    U_deficit_sq_total = np.zeros_like(X_grid)

    plt.close('all')

    # 2. Loop over turbines to accumulate deficit
    for i, t in enumerate(wind_farm.turbines):
        if not t.wake_field:
            continue
            
        # A. Extract the history at Hub Height using existing helpers
        # grid_info gives us the Z-index for hub height
        grid_info = _get_grid_info(t.wake_field[0], t, t.Zhub)
        
        # history gives us 'U' (shape [Ny, N_snapshots]) and 'X' (1D array)
        history = _extract_streamwise_history(t.wake_field, t, grid_info)
        
        if history is None:
            continue
            
        # B. Calculate Local Deficit
        # history['U'] is the absolute velocity.
        U_wake_abs = history['U']
        # Ensure we don't have negative deficits (accelerations) for this visualization
        local_deficit = np.maximum(0, t.Uhub - U_wake_abs)
        
        # C. Map Local Coordinates to Global for Interpolation
        # Turbine Local X -> Global X
        wake_x_global = (history['X'] * t.D) + t.pos[0] 
        # Turbine Local Y -> Global Y
        wake_y_local = grid_info['yloc'][:, 0] * t.D
        wake_y_global = wake_y_local + t.pos[1]
        
        # D. Create Interpolator
        interp = RegularGridInterpolator(
            (wake_y_global, wake_x_global), 
            local_deficit, 
            bounds_error=False, 
            fill_value=0.0
        )
        
        # E. Interpolate onto the Visualization Grid
        pts = np.column_stack((Y_grid.ravel(), X_grid.ravel()))
        deficit_on_grid = interp(pts).reshape(X_grid.shape)
        
        # F. Superposition (Sum of Squares)
        U_deficit_sq_total += deficit_on_grid**2

    # 3. Finalize Total Deficit
    U_deficit_total = np.sqrt(U_deficit_sq_total)
    
    # 4. Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize by the first turbine's Uhub for colorbar
    Uhub_ref = wind_farm.turbines[0]._init_Uhub()
    
    # Plot heatmap
    im = ax.pcolormesh(X_grid, Y_grid, U_deficit_total / Uhub_ref, 
                       cmap='bwr', shading='auto', vmin=0)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'Normalized Velocity Deficit $\Delta U / U_{hub}$')
    
    # Overlay Turbines
    for i, t in enumerate(wind_farm.turbines):
        yaw_rad = np.deg2rad(t.yaw)
        
        # Turbine Blade Line
        dx = (t.D / 2) * np.sin(yaw_rad)
        dy = (t.D / 2) * np.cos(yaw_rad)
        
        ax.plot([t.pos[0] - dx, t.pos[0] + dx], 
                [t.pos[1] + dy, t.pos[1] - dy], 
                color='red', lw=3, solid_capstyle='round')
        
        ax.text(t.pos[0], t.pos[1] + t.D * 0.6, f"T{i}", color='white', 
                ha='center', va='center', fontweight='bold',
                path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])

    ax.set_aspect('equal')
    ax.set_xlabel('Global Streamwise X (m)')
    ax.set_ylabel('Global Cross-stream Y (m)')
    ax.set_title('Wind Farm Velocity Deficit Map (Hub Height)')
    
    if save_path:
        # Ensure directory exists
        if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path) != '':
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved farm map to {save_path}")
    else:
        plt.show()