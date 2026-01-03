from optimization import Optimizer
from config import Config
import numpy as np

def yaw_optimization():
    """Optimize yaw angles for maximum power."""
    print("\n" + "="*70)
    print("Optimizing Yaw Angles")
    print("="*70 + "\n")
    
    config = Config()

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
        max_evals=100,
        acquisition_func='ts',
        log_freq=5,
        verbose=True,
    )

    # Plot results
    optimizer.plot_optimization_history(save_path='Figures/yaw_optimization.png')
    
    # Plot trust region evolution
    optimizer.plot_trust_region_snapshots(
        iterations=[10, 30, 60, 90],
        param_idx_x=0,
        param_idx_y=1,
        save_path='Figures/yaw_trust_region.png'
    )
    
    # Plot GP posterior (works with both TS and EI)
    optimizer.plot_gp_posterior_snapshots(
        iterations=[10, 30, 60, 90],
        param_idx_x=0,
        param_idx_y=1,
        save_path='Figures/yaw_gp_posterior.png'
    )
    
    # Plot acquisition function (only for EI)
    # optimizer.plot_acquisition_function(
    #     iterations=[10, 30, 50, 70],
    #     param_idx_x=0,
    #     param_idx_y=1,
    #     acqf_type='ei',
    #     save_path='Figures/yaw_acquisition.png'
    # )

    optimizer.save_results('Results/yaw_optimization_results.pkl')
    
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
    
    config = Config()
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
    
    optimizer = Optimizer(
        config=config,
        bounds=bounds,
        param_mapping=position_mapping,
        param_names=param_names,
        random_state=42
    )
    
    optimizer.optimize(
        n_init=10,
        max_evals=50,
        acquisition_func='ei',
        verbose=True
    )
    
    optimizer.plot_optimization_history(save_path='Figures/position_optimization.png')
    optimizer.save_results('Results/position_optimization_results.pkl')
    
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
    
    config = Config()
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
    
    optimizer.plot_optimization_history(save_path='Figures/mixed_optimization.png')
    optimizer.save_results('Results/mixed_optimization_results.pkl')
    
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