# Wind Farm Wake Simulation & Optimization

A Python-based framework for simulating wind turbine wakes using vortex particle methods and optimizing wind farm configurations through Bayesian optimization.

## Features

- **Physics-Based Wake Model**: Vortex particle tracking with turbulent viscosity modeling
- **Multiple Wake Superposition**: Momentum-conserving and root-sum-square methods
- **Bayesian Optimization**: Optimize yaw angles, turbine positions, or mixed parameters
- **Advanced Visualization**: 
  - Cross-sectional wake plots with velocity and vorticity fields
  - Streamwise wake evolution panels
  - Optimization convergence and acquisition function surfaces
- **Flexible Configuration**: YAML-based setup with grid initialization
- **Data Management**: Save/load simulation results for post-processing

## Installation

### Requirements

```bash
Python >= 3.8
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.4
pandas >= 1.3
pyyaml >= 5.4
torch >= 2.0
botorch >= 0.9
gpytorch >= 1.11
```

### Setup

```bash
# Clone or download the repository
cd "PVT with Superposition"

# Install dependencies
pip install numpy scipy matplotlib pandas pyyaml
pip install torch botorch gpytorch

# Verify installation
python -c "import numpy, scipy, matplotlib, pandas, yaml, torch, botorch, gpytorch; print('All dependencies installed!')"
```

## Quick Start

### 1. Basic Simulation

```python
from config import Config
from simulation import Simulation

# Create default configuration (single turbine)
config = Config()

# Run simulation
sim = Simulation(config)
sim.run()

# Calculate total power output
total_power = sim.calculate_objective()
print(f"Total Power: {total_power/1e6:.2f} MW")

# Save results
sim.save_results(out_path="Data/")
```

### 2. Multi-Turbine Wind Farm

```python
from config import Config, WindFarmConfig
import numpy as np

# Define wind farm configuration
wf_config = WindFarmConfig(
    grid=(2, 3, 5, 5),  # 2 rows × 3 columns, 5D spacing
    D=np.array([126.0]),  # Rotor diameter (m)
    Zhub=np.array([90.0]),  # Hub height (m)
    Ct=np.array([0.8]),  # Thrust coefficient
    yaw=np.array([0.0, 10.0, 0.0, -10.0, 0.0, 0.0])  # Individual yaw angles
)

config = Config(WindFarm=wf_config)

sim = Simulation(config)
sim.run()
```

### 3. Yaw Angle Optimization

```python
from run_optimization import optimize_yaw_angles
from config import Config

config = Config()

# Optimize yaw angles using TuRBO with Thompson Sampling
optimizer = optimize_yaw_angles(
    config=config,
    yaw_bounds=[(-30, 30)] * len(config.WindFarm),
    acquisition_func='ts',  # 'ts' (Thompson Sampling) or 'ei' (Expected Improvement)
    n_init=20,
    max_evals=100,
    random_state=42,
    verbose=True
)

# Plot results
optimizer.plot_optimization_history(save_path='optimization_history.png')
optimizer.plot_trust_region_snapshots(
    iterations=[10, 30, 50, 70],
    save_path='trust_region_evolution.png'
)
optimizer.plot_gp_posterior_snapshots(
    iterations=[10, 30, 50, 70],
    save_path='gp_posterior.png'
)
```

## Configuration Guide

### Wind Farm Setup

#### Option 1: Grid-Based Layout

```python
from config import Config, WindFarmConfig

wf_config = WindFarmConfig(
    grid=(3, 3, 7, 5),  # rows, cols, x-spacing, y-spacing
    dist_type='D',      # Spacing in rotor diameters
    D=np.array([126.0]),
    Zhub=np.array([90.0]),
    Ct=np.array([0.8]),
    yaw=np.array([0.0])  # Applied to all turbines
)

config = Config(WindFarm=wf_config)
```

#### Option 2: Manual Positions

```python
wf_config = WindFarmConfig(
    pos=np.array([
        [0.0, 0.0, 0.0],      # Turbine 1
        [630.0, 0.0, 0.0],    # Turbine 2 (5D downstream)
        [1260.0, 0.0, 0.0]    # Turbine 3 (10D downstream)
    ]),
    D=np.array([126.0, 126.0, 126.0]),
    yaw=np.array([15.0, 0.0, -10.0])
)
```

#### Option 3: Complex Terrain

```python
def elevation_func(x, y):
    """Custom terrain elevation function"""
    return 0.1 * np.sin(x / 200) + 0.05 * np.cos(y / 150)

wf_config = WindFarmConfig(
    grid=(2, 4, 5, 5),
    elevation_func=elevation_func  # Auto-applies to z-coords
)
```

### Field Parameters

```python
from config import FieldConfig

field_config = FieldConfig(
    Uh=8.55,        # Hub-height wind speed (m/s)
    Zh=80.0,        # Measurement height (m)
    I_amb=0.072,    # Ambient turbulence intensity
    z0=0.03,        # Surface roughness (m)
    max_X=15.0,     # Downstream extent (in D)
    max_Y=3.0,      # Lateral extent (in D)
    max_Z=2.0,      # Vertical extent (in D)
    n_grids=20      # Grid resolution
)

config = Config(WindFarm=wf_config, Field=field_config)
```

### Save/Load Configurations

```python
# Save configuration
config.save_yaml('my_windfarm.yaml')

# Load configuration
config = Config.load_yaml('my_windfarm.yaml')
```

## Visualization

### Wake Field Plots

```python
from plotting import plot_yaw_sweep_panel, plot_wake_evolution_panel

# Cross-sectional snapshots at different x/D positions
plot_yaw_sweep_panel(
    data_path="Data/",
    turbine_idx=0,
    yaw_angles=[0.0, 15.0, 30.0],    # Rows
    x_positions=[2, 4, 6, 8],         # Columns
    D=126.0,
    Uhub=8.55,
    plot_type='velocity',             # or 'vorticity'
    save_path='Figures/yaw_sweep.png'
)

# Streamwise wake evolution
plot_wake_evolution_panel(
    data_path="Data/",
    turbine_idx=0,
    yaw_angles=[0.0, 15.0, 30.0],
    X_limit=10.0,                     # Max x/D to plot
    Z_hub=90.0,                       # Hub height for slice
    draw_centerline=True,             # Track wake center
    save_path='Figures/wake_evolution.png'
)
```

### Load and Plot Saved Results

```python
from plotting import load_saved_results, plot_velocity_contour
import matplotlib.pyplot as plt

# Load results
results = load_saved_results("Data/")
frames = results[0]['frames']  # Turbine 0

# Setup grid
frame0 = frames[0]
grid = {
    'yloc': frame0['yloc'] / 126.0,
    'zloc': frame0['zloc'] / 126.0,
    'Ny': frame0['yloc'].shape[0],
    'Nz': frame0['yloc'].shape[1]
}

# Plot specific frame
fig, ax = plt.subplots()
state = {}
levels = {'u': np.linspace(0, 1, 21)}
plot_velocity_contour(ax, state, frames[10], grid, levels, D=126.0, Uhub=8.55)
plt.show()
```

## Optimization Workflows

### 1. Yaw Angle Optimization

```python
from run_optimization import optimize_yaw_angles

optimizer = optimize_yaw_angles(
    config=config,
    yaw_bounds=[(-30, 30)] * n_turbines,
    acquisition_func='ts',  # 'ts' (Thompson Sampling) or 'ei' (Expected Improvement)
    n_init=20,              # Initial Sobol samples
    max_evals=100,          # Maximum function evaluations
    random_state=42,
    device='cpu',           # 'cpu' or 'cuda'
    verbose=True
)

# Access results
best_yaws = optimizer.best_params
best_power = optimizer.best_value
```

### 2. Turbine Layout Optimization

```python
from run_optimization import position_optimization

# Optimizes x,y positions for all turbines
optimizer = position_optimization()

# Or use custom bounds with Optimizer class directly
from optimization import Optimizer
import numpy as np

bounds = []
for i in range(n_turbines):
    bounds.append([0, 2000])      # x bounds
    bounds.append([-500, 500])    # y bounds

optimizer = Optimizer(
    config=config,
    bounds=np.array(bounds),
    param_mapping=position_mapping_func,
    param_names=[f'x_T{i}' or f'y_T{i}' for i in range(n_turbines)]
)

optimizer.optimize(n_init=30, max_evals=150)
```

### 3. Mixed Parameter Optimization

```python
from run_optimization import mixed_optimization, custom_optimization

# Pre-built example: yaw + thrust coefficient
optimizer = mixed_optimization()

# Or custom: yaw + hub height
optimizer = custom_optimization()

# Or define your own with Optimizer class
from optimization import Optimizer

def custom_mapping(params, cfg):
    cfg.WindFarm.yaw = params[:n_turbines]
    cfg.WindFarm.Ct[0] = params[-1]
    return cfg

bounds = [[-30, 30]] * n_turbines + [[0.6, 0.9]]  # yaw angles + Ct

optimizer = Optimizer(
    config=config,
    bounds=np.array(bounds),
    param_mapping=custom_mapping
)
optimizer.optimize(max_evals=80)
```

### Advanced Optimization Plots

```python
# Optimization convergence
optimizer.plot_optimization_history('opt_history.png')

# Trust region evolution (works with TS and EI)
optimizer.plot_trust_region_snapshots(
    iterations=[10, 30, 50, 70],
    param_idx_x=0,  # Turbine 0 yaw
    param_idx_y=1,  # Turbine 1 yaw
    save_path='trust_region_evolution.png'
)

# GP posterior mean and uncertainty
optimizer.plot_gp_posterior_snapshots(
    iterations=[10, 30, 50, 70],
    param_idx_x=0,
    param_idx_y=1,
    save_path='gp_posterior.png'
)

# Acquisition function surface (only for EI, not TS)
optimizer.plot_acquisition_function(
    iterations=[10, 30, 50, 70],
    param_idx_x=0,
    param_idx_y=1,
    acqf_type='ei',
    save_path='ei_surface.png'
)
```

## File Structure

```
PVT with Superposition/
├── config.py                   # Configuration dataclasses
├── run_simulation.py           # Main simulation script
├── run_optimization.py         # Optimization script
├── optimization.py             # Bayesian optimization framework
├── plotting.py                 # Visualization tools
├── simulation/
│   ├── __init__.py            # Simulation class
│   ├── core_types.py          # Turbine and WindFarm classes
│   ├── vortex_model.py        # Vortex particle tracking
│   ├── model_solver.py        # Wake field solver
│   ├── superposition.py       # Multi-wake superposition
│   ├── data_structures.py     # Data containers
│   └── utils.py               # Utility functions
├── Data/                       # Saved simulation results
└── Figures/                    # Generated plots
```

## API Reference

### Core Classes

#### `Config`
Main configuration container
- `WindFarm`: Wind farm layout and turbine properties
- `Field`: Flow field and simulation parameters
- `save_yaml(path)`: Save configuration
- `load_yaml(path)`: Load configuration

#### `Simulation`
Main simulation driver
- `run()`: Execute wake simulation
- `calculate_objective()`: Compute total power output
- `save_results(out_path, limit_frames)`: Save wake field data

#### `Optimizer`
Trust Region Bayesian Optimization (TuRBO) framework
- `optimize(n_init, max_evals, acquisition_func, log_freq)`: Run optimization with adaptive trust regions
- `plot_optimization_history()`: Plot convergence and parameter evolution
- `plot_trust_region_snapshots()`: Visualize trust region adaptation
- `plot_acquisition_function()`: Show EI surface (EI only)
- `plot_gp_posterior_snapshots()`: Show GP posterior mean and uncertainty
- `save_results(filepath)`: Save optimization history to pickle file

### Visualization Functions

- `plot_yaw_sweep_panel()`: Cross-sections at multiple yaw angles and x/D positions
- `plot_wake_evolution_panel()`: Streamwise wake development
- `plot_farm_deficit_map()`: Top and side views of entire farm
- `load_saved_results()`: Load CSV simulation data

## Advanced Features

### Custom Objective Functions

```python
from optimization import Optimizer
import copy

def custom_objective(params, base_config):
    """Maximize power while penalizing yaw magnitude"""
    temp_config = copy.deepcopy(base_config)
    temp_config.WindFarm.yaw = params
    
    sim = Simulation(temp_config)
    sim.run()
    
    power = sim.calculate_objective()
    yaw_penalty = np.sum(np.abs(params)) * 1e4
    
    return power - yaw_penalty

# Use custom optimizer
optimizer = Optimizer(
    config=config,
    bounds=np.array([(-30, 30)] * n_turbines),
    param_mapping=lambda params, cfg: setattr(cfg.WindFarm, 'yaw', params) or cfg,
    param_names=[f'yaw_T{i}' for i in range(n_turbines)]
)

# Replace objective function
optimizer.objective_function = lambda params: custom_objective(
    optimizer._denormalize(params), config
)

optimizer.optimize(n_iter=50)
```

### Parallel Evaluation (Future Work)

```python
# TODO: Implement parallel objective evaluation
from multiprocessing import Pool

def parallel_optimization(n_workers=4):
    # Evaluate multiple candidates simultaneously
    pass
```

## Performance Tips

1. **Grid Resolution**: Start with `n_grids=15` for quick tests, use `n_grids=25` for production
2. **Frame Limiting**: Use `save_results(limit_frames=50)` to reduce storage
3. **Initial Sampling**: Use `n_init=max(10, 2*n_params)` for Sobol initialization
4. **Acquisition Function**: 
   - `'ts'` (Thompson Sampling): Better for high-dimensional problems, more exploration
   - `'ei'` (Expected Improvement): Better for low-dimensional problems, more exploitation
5. **Trust Region**: Automatically adapts - expands on success, shrinks on failure
6. **GPU Acceleration**: Set `device='cuda'` if available (optimizer runs on GPU, simulation on CPU)
7. **Logging**: Adjust `log_freq` parameter to control console output frequency
```

## Contributing

Contributions are welcome! Areas for improvement:
- Parallel objective evaluation
- Wind rose optimization
- Real-time visualization
- GPU acceleration for large farms

## References

1. Vortex Particle Methods for wake modeling
2. Trust Region Bayesian Optimization (TuRBO) - Eriksson et al. (2019)
3. Thompson Sampling for Bayesian Optimization - Hernández-Lobato et al. (2014)
4. Momentum-Conserving Superposition (MCS) for multiple wakes
5. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization

## License

This work is licensed under MIT license.

## Contact

For bug reports, feature requests, or code-related issues, please [open an issue](https://github.com/alifuatsahin/Wind-Farm-Optimization/issues).

---

**Note**: This is a research-grade simulation tool. Validate results against experimental data or higher-fidelity CFD for production applications.
