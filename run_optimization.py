import os

from optimization import Optimizer, yaw_parameterisation
from config import Config
import numpy as np

def yaw_optimization(parameterisation="per_rotor", tol_D=0.5):
    """Optimize yaw angles for maximum power.

    parameterisation='per_rotor' gives every rotor its own angle. That is the general
    case and the one to keep for wind that is not perpendicular to the rows, since the
    station grouping then ties together rotors that no longer share an inflow.

    'station' gives one angle per streamwise station. On the 3x3 that reached a better
    optimum than per-rotor in 18 evaluations against 140 -- not because the restriction
    helps in principle (it is a strict subspace) but because a per-rotor search does not
    converge at any budget a large farm can afford. It is what makes an 80-rotor farm a
    10-parameter problem rather than an 80-parameter one.
    """
    print("\n" + "="*70)
    print("Optimizing Yaw Angles")
    print("="*70 + "\n")

    config = Config(run_prefix="opt")

    pos = np.array([t.pos[:2] for t in config.WindFarm.turbines()], dtype=float)
    D_ref = float(config.WindFarm.turbines()[0].D)
    to_yaw, n_params, gid = yaw_parameterisation(pos, D_ref, parameterisation, tol_D)
    if gid is None:
        print(f"Parameterisation: per-rotor, {n_params} free angles\n")
        names = [f'yaw_T{i}' for i in range(n_params)]
    else:
        print(f"Parameterisation: station-wise, {n_params} free angles for "
              f"{len(pos)} rotors (station width {tol_D:g} D)")
        for g in range(n_params):
            members = np.flatnonzero(gid == g)
            print(f"   station {g}: x = {pos[members[0], 0] / D_ref:6.2f} D, "
                  f"rotors {list(members)}")
        print("")
        names = [f'yaw_S{g}' for g in range(n_params)]

    def yaw_mapping(params, cfg):
        cfg.WindFarm.yaw = to_yaw(params)
        return cfg

    optimizer = Optimizer(
        config=config,
        bounds=[(-30, 30)] * n_params,
        device='cuda',
        param_mapping=yaw_mapping,
        param_names=names,
        random_state=42
    )

    optimizer.optimize(
        n_init=20,
        max_evals=300,
        acquisition_func='ts',
        log_freq=5,
        verbose=True,
    )

    # Plot results
    optimizer.plot_optimization_history(save_path=os.path.join(config.out_path, 'figures', 'yaw_optimization.png'))

    # Plot trust region evolution
    optimizer.plot_trust_region_snapshots(
        iterations=[10, 30, 60, 90],
        param_idx_x=0,
        param_idx_y=1,
        save_path=os.path.join(config.out_path, 'figures', 'yaw_trust_region.png')
    )

    # Plot GP posterior (works with both TS and EI)
    optimizer.plot_gp_posterior_snapshots(
        iterations=[10, 30, 60, 90],
        param_idx_x=0,
        param_idx_y=1,
        save_path=os.path.join(config.out_path, 'figures', 'yaw_gp_posterior.png')
    )

    # Plot acquisition function (only for EI)
    # optimizer.plot_acquisition_function(
    #     iterations=[10, 30, 50, 70],
    #     param_idx_x=0,
    #     param_idx_y=1,
    #     acqf_type='ei',
    #     save_path=os.path.join(config.out_path, 'figures', 'yaw_acquisition.png')
    # )

    optimizer.save_results(os.path.join(config.out_path, 'results', 'yaw_optimization_results.pkl'))
    
    print("\n" + "="*70)
    print("RESULTS: Best yaw configuration found")
    print("="*70)
    for i, yaw in enumerate(optimizer.best_params):
        print(f"  Turbine {i}: {yaw:.2f}°")
    print(f"\nWind Farm Efficiency: {optimizer.best_value:.2f}")
    
    return optimizer


def position_optimization():
    """Optimize turbine x,y positions for maximum power."""
    print("\n" + "="*70)
    print("Optimizing Turbine Positions")
    print("="*70 + "\n")

    config = Config(run_prefix="opt")
    n_turbines = len(config.WindFarm)
    
    # Define bounds: [x0, y0, x1, y1, ...]
    bounds = []
    param_names = []
    for i in range(n_turbines):
        bounds.append([0, 1000])      # x position (m)
        bounds.append([-200, 200])    # y position (m)
        param_names.extend([f'x_T{i}', f'y_T{i}'])
    
    bounds = np.array(bounds)
    
    def position_mapping(params, cfg):
        # Update turbine positions (keep z unchanged)
        new_pos = cfg.WindFarm.pos.copy()
        for i in range(n_turbines):
            new_pos[i, 0] = params[2*i]      # x
            new_pos[i, 1] = params[2*i + 1]  # y
        cfg.WindFarm.pos = new_pos
        return cfg

    # Minimum separation. Without it the optimiser is free to "win" with layouts that
    # cannot be built: replaying 14 Sobol draws over these bounds, the closest pair came
    # to 0.23 D -- discs essentially coincident -- and 13 of the 14 were under 3 D, with
    # the model returning a plausible-looking efficiency and raising nothing. It is a
    # formulation problem rather than a robustness one, and it matters most because
    # overlapping rotors are exactly where this model has no physics: there is no
    # blockage, and the station-grouping and ramp-stacking both assume rotors that do
    # not sit inside each other's near wake.
    MIN_SEP_D = 3.0
    D_ref = float(np.atleast_1d(config.WindFarm.D)[0])

    def min_separation_ok(params):
        xy = np.column_stack([params[0::2], params[1::2]])
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                if np.hypot(*(xy[i] - xy[j])) < MIN_SEP_D * D_ref:
                    return False
        return True

    print(f"Constraint: pairwise separation >= {MIN_SEP_D:g} D "
          f"({MIN_SEP_D * D_ref:.0f} m)\n")

    optimizer = Optimizer(
        config=config,
        bounds=bounds,
        param_mapping=position_mapping,
        param_names=param_names,
        random_state=42,
        feasible=min_separation_ok,
    )
    
    optimizer.optimize(
        n_init=10,
        max_evals=50,
        acquisition_func='ei',
        verbose=True
    )
    
    optimizer.plot_optimization_history(save_path=os.path.join(config.out_path, 'figures', 'position_optimization.png'))
    optimizer.save_results(os.path.join(config.out_path, 'results', 'position_optimization_results.pkl'))
    
    print("\n" + "="*70)
    print("RESULTS: Best position configuration found")
    print("="*70)
    for i in range(n_turbines):
        x = optimizer.best_params[2*i]
        y = optimizer.best_params[2*i + 1]
        print(f"  Turbine {i}: x={x:.2f} m, y={y:.2f} m")
    print(f"\nTotal Power: {optimizer.best_value:.2f} W")
    
    return optimizer


def mixed_optimization():
    """Optimize mixed parameters (yaw + thrust coefficient)."""
    print("\n" + "="*70)
    print("Optimizing Mixed Parameters (Yaw + Ct)")
    print("="*70 + "\n")

    config = Config(run_prefix="opt")
    n_turbines = len(config.WindFarm)
    
    # Define bounds: [yaw_0, yaw_1, ..., Ct_0]
    bounds = []
    param_names = []
    
    # Yaw angles for all turbines
    for i in range(n_turbines):
        bounds.append([-30, 30])
        param_names.append(f'yaw_T{i}')
    
    # Thrust coefficient for first turbine
    bounds.append([0.6, 0.9])
    param_names.append('Ct_T0')
    
    bounds = np.array(bounds)
    
    def mixed_mapping(params, cfg):
        # First n_turbines params are yaw angles
        cfg.WindFarm.yaw = params[:n_turbines].copy()
        
        # Last param is Ct for turbine 0
        new_ct = cfg.WindFarm.Ct.copy()
        new_ct[0] = params[-1]
        cfg.WindFarm.Ct = new_ct
        
        return cfg
    
    optimizer = Optimizer(
        config=config,
        bounds=bounds,
        param_mapping=mixed_mapping,
        param_names=param_names,
        random_state=42
    )
    
    optimizer.optimize(
        n_init=15,
        max_evals=60,
        acquisition_func='ts',
        verbose=True
    )
    
    optimizer.plot_optimization_history(save_path=os.path.join(config.out_path, 'figures', 'mixed_optimization.png'))
    optimizer.save_results(os.path.join(config.out_path, 'results', 'mixed_optimization_results.pkl'))
    
    print("\n" + "="*70)
    print("RESULTS: Best parameter configuration found")
    print("="*70)
    for name, value in zip(optimizer.param_names, optimizer.best_params):
        print(f"  {name}: {value:.2f}")
    print(f"\nTotal Power: {optimizer.best_value:.2f} W")
    
    return optimizer

if __name__ == "__main__":    
    # Run yaw optimization
    optimizer = yaw_optimization()
    
    # Uncomment to run other examples:
    # optimizer = position_optimization()
    # optimizer = mixed_optimization()