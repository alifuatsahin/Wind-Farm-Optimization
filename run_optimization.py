from optimization import (
    optimize_yaw_angles, 
    optimize_turbine_positions, 
    optimize_mixed_params,
    Optimizer
)
from config import Config
import numpy as np


def example_1_yaw_optimization():
    """Example 1: Optimize yaw angles for maximum power."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Optimizing Yaw Angles")
    print("="*70 + "\n")
    
    config = Config()
    
    # Optimize yaw angles with custom bounds for each turbine
    optimizer = optimize_yaw_angles(
        config=config,
        yaw_bounds=[(-30, 30)] * len(config.WindFarm),  # Same bounds for all turbines
        acquisition_func='ei',
        xi=0.01,
        random_state=42,
        n_iter=15,
        n_init=5,
        verbose=True
    )
    
    optimizer.plot_optimization_history(save_path='yaw_optimization.png')
    
    print("\n" + "="*70)
    print("RESULTS: Best yaw configuration found")
    print("="*70)
    for i, yaw in enumerate(optimizer.best_params):
        print(f"  Turbine {i}: {yaw:.2f}°")
    print(f"\nTotal Power: {optimizer.best_value:.2f} W")
    
    return optimizer


def example_2_position_optimization():
    """Example 2: Optimize turbine x,y positions."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Optimizing Turbine Positions")
    print("="*70 + "\n")
    
    config = Config()
    n_turbines = len(config.WindFarm)
    
    # Define search space for positions
    optimizer = optimize_turbine_positions(
        config=config,
        x_bounds=[(0, 500)] * n_turbines,      # Streamwise position
        y_bounds=[(-200, 200)] * n_turbines,   # Lateral position
        acquisition_func='ei',
        xi=0.01,
        random_state=42,
        n_iter=15,
        n_init=5,
        verbose=True
    )
    
    optimizer.plot_optimization_history(save_path='position_optimization.png')
    
    print("\n" + "="*70)
    print("RESULTS: Best position configuration found")
    print("="*70)
    for i in range(n_turbines):
        x = optimizer.best_params[2*i]
        y = optimizer.best_params[2*i + 1]
        print(f"  Turbine {i}: x={x:.2f} m, y={y:.2f} m")
    print(f"\nTotal Power: {optimizer.best_value:.2f} W")
    
    return optimizer


def example_3_mixed_optimization():
    """Example 3: Optimize mixed parameters (yaw + thrust coefficient)."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Optimizing Mixed Parameters (Yaw + Ct)")
    print("="*70 + "\n")
    
    config = Config()
    n_turbines = len(config.WindFarm)
    
    # Define which parameters to optimize
    param_specs = {}
    
    # Add yaw for each turbine
    for i in range(n_turbines):
        param_specs[f'yaw_T{i}'] = {
            'bounds': (-30, 30),
            'indices': [i],
            'path': 'WindFarm.yaw'
        }
    
    # Add thrust coefficient for first turbine
    param_specs['Ct_T0'] = {
        'bounds': (0.6, 0.9),
        'indices': [0],
        'path': 'WindFarm.Ct'
    }
    
    optimizer = optimize_mixed_params(
        config=config,
        param_specs=param_specs,
        acquisition_func='ei',
        xi=0.01,
        random_state=42,
        n_iter=15,
        n_init=5,
        verbose=True
    )
    
    optimizer.plot_optimization_history(save_path='mixed_optimization.png')
    
    print("\n" + "="*70)
    print("RESULTS: Best parameter configuration found")
    print("="*70)
    for name, value in zip(optimizer.param_names, optimizer.best_params):
        print(f"  {name}: {value:.2f}")
    print(f"\nTotal Power: {optimizer.best_value:.2f} W")
    
    return optimizer


def example_4_custom_optimization():
    """Example 4: Custom optimization with manual parameter mapping."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Optimization (Yaw + Hub Height)")
    print("="*70 + "\n")
    
    config = Config()
    n_turbines = len(config.WindFarm)
    
    # Define custom bounds: [yaw_T0, yaw_T1, ..., Zhub_T0]
    bounds = []
    param_names = []
    
    for i in range(n_turbines):
        bounds.append([-30, 30])
        param_names.append(f'yaw_T{i}')
    
    bounds.append([70, 90])  # Hub height for first turbine
    param_names.append('Zhub_T0')
    
    bounds = np.array(bounds)
    
    # Define custom parameter mapping
    def custom_mapping(params, cfg):
        # First n_turbines params are yaw angles
        cfg.WindFarm.yaw = params[:n_turbines].copy()
        
        # Last param is hub height for turbine 0
        new_zhub = cfg.WindFarm.Zhub.copy()
        new_zhub[0] = params[-1]
        cfg.WindFarm.Zhub = new_zhub
        
        return cfg
    
    optimizer = Optimizer(
        config=config,
        bounds=bounds,
        param_mapping=custom_mapping,
        param_names=param_names,
        acquisition_func='ei',
        xi=0.01,
        random_state=42
    )
    
    best_params, best_value = optimizer.optimize(
        n_iter=15,
        n_init=5,
        verbose=True
    )
    
    optimizer.plot_optimization_history(save_path='custom_optimization.png')
    
    print("\n" + "="*70)
    print("RESULTS: Best custom configuration found")
    print("="*70)
    for name, value in zip(param_names, best_params):
        print(f"  {name}: {value:.2f}")
    print(f"\nTotal Power: {best_value:.2f} W")
    
    return optimizer


if __name__ == "__main__":
    # Choose which example to run
    print("Select optimization example:")
    print("1. Yaw angles only")
    print("2. Turbine positions")
    print("3. Mixed parameters (yaw + Ct)")
    print("4. Custom (yaw + hub height)")
    
    # Run example 1 by default (comment/uncomment as needed)
    optimizer = example_1_yaw_optimization()
    
    # Uncomment to run other examples:
    # optimizer = example_2_position_optimization()
    # optimizer = example_3_mixed_optimization()
    # optimizer = example_4_custom_optimization()
