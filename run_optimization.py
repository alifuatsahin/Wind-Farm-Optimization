import os

from optimization import Optimizer
from config import Config
import numpy as np

def yaw_optimization():
    """Optimize yaw angles for maximum power."""
    print("\n" + "="*70)
    print("Optimizing Yaw Angles")
    print("="*70 + "\n")

    config = Config(run_prefix="opt")

    def yaw_mapping(params, cfg):
        cfg.WindFarm.yaw = params.copy()
        return cfg

    optimizer = Optimizer(
        config=config,
        bounds=[(-30, 30)] * len(config.WindFarm),
        device='cuda',
        param_mapping=yaw_mapping,
        param_names=[f'yaw_T{i}' for i in range(len(config.WindFarm))],
        random_state=42
    )

    optimizer.optimize(
        n_init=20,
        max_evals=300,
        acquisition_func='ts',
        log_freq=5,
        verbose=True,
    )

    optimizer.plot_optimization_history(save_path=os.path.join(config.out_path, 'figures', 'yaw_optimization.png'))

    optimizer.plot_trust_region_snapshots(
        iterations=[10, 30, 60, 90],
        param_idx_x=0,
        param_idx_y=1,
        save_path=os.path.join(config.out_path, 'figures', 'yaw_trust_region.png')
    )

    optimizer.plot_gp_posterior_snapshots(
        iterations=[10, 30, 60, 90],
        param_idx_x=0,
        param_idx_y=1,
        save_path=os.path.join(config.out_path, 'figures', 'yaw_gp_posterior.png')
    )


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

    bounds = []
    param_names = []
    for i in range(n_turbines):
        bounds.append([0, 1000])
        bounds.append([-200, 200])
        param_names.extend([f'x_T{i}', f'y_T{i}'])

    bounds = np.array(bounds)

    def position_mapping(params, cfg):
        new_pos = cfg.WindFarm.pos.copy()
        for i in range(n_turbines):
            new_pos[i, 0] = params[2*i]
            new_pos[i, 1] = params[2*i + 1]
        cfg.WindFarm.pos = new_pos
        return cfg

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

    bounds = []
    param_names = []

    for i in range(n_turbines):
        bounds.append([-30, 30])
        param_names.append(f'yaw_T{i}')

    bounds.append([0.6, 0.9])
    param_names.append('Ct_T0')

    bounds = np.array(bounds)

    def mixed_mapping(params, cfg):
        cfg.WindFarm.yaw = params[:n_turbines].copy()

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
    optimizer = yaw_optimization()
