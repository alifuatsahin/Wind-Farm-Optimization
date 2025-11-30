import numpy as np
from config import Config, WindFarmConfig
from simulation import Simulation
from typing import Callable, Tuple, Optional, List, Dict, Any
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy


class Optimizer:
    """
    Generic Bayesian Optimization for wind farm configuration optimization.
    
    Uses Gaussian Process regression to model the objective function
    and acquisition functions to balance exploration and exploitation.
    
    Can optimize any parameters in the config (yaw angles, positions, Ct, etc.)
    by providing a custom parameter mapping function.
    """
    
    def __init__(
        self,
        config: Config,
        bounds: np.ndarray,
        param_mapping: Callable[[np.ndarray, Config], Config],
        param_names: Optional[List[str]] = None,
        acquisition_func: str = 'ei',
        xi: float = 0.01,
        random_state: Optional[int] = None
    ):
        """
        Initialize the Bayesian Optimizer.
        
        Parameters:
        -----------
        config : Config
            Base configuration for the wind farm
        bounds : np.ndarray
            Bounds for optimization parameters, shape (n_params, 2) with [min, max] for each parameter
        param_mapping : Callable[[np.ndarray, Config], Config]
            Function that takes optimization parameters and base config, returns modified config
            Example: lambda params, cfg: set_yaw_angles(params, cfg)
        param_names : List[str], optional
            Names of parameters being optimized (for plotting/logging)
        acquisition_func : str
            Acquisition function: 'ei' (Expected Improvement), 'pi' (Probability of Improvement), 'ucb' (Upper Confidence Bound)
        xi : float
            Exploration-exploitation trade-off parameter
        random_state : int, optional
            Random seed for reproducibility
        """
        self.config = config
        self.bounds = np.asarray(bounds)
        self.n_params = len(bounds)
        self.param_mapping = param_mapping
        self.param_names = param_names if param_names is not None else [f'param_{i}' for i in range(self.n_params)]
        self.acquisition_func = acquisition_func.lower()
        self.xi = xi
        self.rng = np.random.default_rng(random_state)
        
        # Storage for optimization history
        self.X_samples = []  # Evaluated parameter configurations
        self.y_samples = []  # Corresponding objective values
        self.best_value = -np.inf
        self.best_params = None
        
    def objective_function(self, params: np.ndarray) -> float:
        """
        Evaluate the objective function for given parameters.
        
        Parameters:
        -----------
        params : np.ndarray
            Optimization parameters
            
        Returns:
        --------
        float
            Objective value (higher is better)
        """
        # Apply parameter mapping to create config
        temp_config = self.param_mapping(params, copy.deepcopy(self.config))
        
        # Run simulation
        sim = Simulation(temp_config)
        sim.run()
        
        # Calculate total power (default objective)
        total_power = sim.wind_farm.calculate_power_output()
        
        return total_power
    
    def _fit_gp(self):
        """Fit Gaussian Process to current samples."""
        
        X = np.array(self.X_samples)
        y = np.array(self.y_samples).reshape(-1, 1)
        
        # Define kernel: Matern kernel with constant scaling
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(self.n_params),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5
        )
        
        # Fit GP
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=self.rng.integers(0, 1e6)
        )
        gp.fit(X, y)
        
        return gp
    
    def _acquisition(self, X: np.ndarray, gp, y_max: float) -> np.ndarray:
        """
        Calculate acquisition function values.
        
        Parameters:
        -----------
        X : np.ndarray
            Candidate points, shape (n_samples, n_params)
        gp : GaussianProcessRegressor
            Fitted Gaussian Process
        y_max : float
            Current best observed value
            
        Returns:
        --------
        np.ndarray
            Acquisition function values
        """
        
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.flatten()
        sigma = sigma.flatten()
        
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)
        
        if self.acquisition_func == 'ei':
            # Expected Improvement
            Z = (mu - y_max - self.xi) / sigma
            ei = (mu - y_max - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            return ei
        
        elif self.acquisition_func == 'pi':
            # Probability of Improvement
            Z = (mu - y_max - self.xi) / sigma
            return norm.cdf(Z)
        
        elif self.acquisition_func == 'ucb':
            # Upper Confidence Bound
            kappa = 2.576  # 99% confidence
            return mu + kappa * sigma
        
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_func}")
    
    def _propose_location(self, gp, n_random: int = 10000) -> np.ndarray:
        """
        Propose next sampling point by optimizing acquisition function.
        
        Parameters:
        -----------
        gp : GaussianProcessRegressor
            Fitted Gaussian Process
        n_random : int
            Number of random samples for acquisition optimization
            
        Returns:
        --------
        np.ndarray
            Proposed parameter values
        """
        # Generate random candidates within bounds
        dim = self.n_params
        X_random = np.zeros((n_random, dim))
        for i in range(dim):
            X_random[:, i] = self.rng.uniform(
                self.bounds[i, 0],
                self.bounds[i, 1],
                n_random
            )
        
        # Evaluate acquisition function
        y_max = np.max(self.y_samples)
        acq_values = self._acquisition(X_random, gp, y_max)
        
        # Return point with highest acquisition value
        best_idx = np.argmax(acq_values)
        return X_random[best_idx]
    
    def _latin_hypercube_sampling(self, n_samples: int) -> np.ndarray:
        """
        Generate initial samples using Latin Hypercube Sampling.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        np.ndarray
            Initial sample points, shape (n_samples, n_params)
        """
        dim = self.n_params
        samples = np.zeros((n_samples, dim))
        
        for i in range(dim):
            # Divide range into n_samples intervals
            intervals = np.linspace(0, 1, n_samples + 1)
            # Sample uniformly within each interval
            samples[:, i] = self.rng.uniform(intervals[:-1], intervals[1:])
            # Shuffle
            self.rng.shuffle(samples[:, i])
            # Scale to bounds
            samples[:, i] = self.bounds[i, 0] + samples[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
        
        return samples
    
    def optimize(
        self,
        n_iter: int = 20,
        n_init: int = 5,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Run Bayesian optimization.
        
        Parameters:
        -----------
        n_iter : int
            Number of optimization iterations
        n_init : int
            Number of initial random samples
        verbose : bool
            Print progress information
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            Best parameters and corresponding objective value
        """
        # Initial sampling using Latin Hypercube
        if verbose:
            print(f"Starting Bayesian Optimization")
            print(f"Number of parameters: {self.n_params}")
            print(f"Parameter names: {self.param_names}")
            print(f"Bounds: {self.bounds}")
            print(f"\nInitial sampling phase ({n_init} samples)...")
        
        X_init = self._latin_hypercube_sampling(n_init)
        
        for i, x in enumerate(X_init):
            start_time = time.time()
            y = self.objective_function(x)
            eval_time = time.time() - start_time
            
            self.X_samples.append(x)
            self.y_samples.append(y)
            
            if y > self.best_value:
                self.best_value = y
                self.best_params = x.copy()
            
            if verbose:
                param_str = ', '.join([f'{name}={val:.2f}' for name, val in zip(self.param_names, x)])
                print(f"  Init {i+1}/{n_init}: [{param_str}], Value={y:.2f}, Time={eval_time:.1f}s")
        
        # Bayesian optimization iterations
        if verbose:
            print(f"\nOptimization phase ({n_iter} iterations)...")
        
        for i in range(n_iter):
            # Fit GP to current data
            gp = self._fit_gp()
            
            # Propose next point
            x_next = self._propose_location(gp)
            
            # Evaluate objective
            start_time = time.time()
            y_next = self.objective_function(x_next)
            eval_time = time.time() - start_time
            
            # Update samples
            self.X_samples.append(x_next)
            self.y_samples.append(y_next)
            
            # Update best
            if y_next > self.best_value:
                self.best_value = y_next
                self.best_params = x_next.copy()
                improvement = " *** NEW BEST ***"
            else:
                improvement = ""
            
            if verbose:
                param_str = ', '.join([f'{name}={val:.2f}' for name, val in zip(self.param_names, x_next)])
                print(f"  Iter {i+1}/{n_iter}: [{param_str}], Value={y_next:.2f}, Time={eval_time:.1f}s {improvement}")
        
        if verbose:
            print(f"\nOptimization complete!")
            for name, val in zip(self.param_names, self.best_params):
                print(f"  {name}: {val:.2f}")
            print(f"Best objective value: {self.best_value:.2f}")
        
        return self.best_params, self.best_value
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot the optimization history.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """        
        iterations = np.arange(len(self.y_samples))
        y_best_so_far = np.maximum.accumulate(self.y_samples)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot objective value over iterations
        ax1.plot(iterations, self.y_samples, 'bo-', alpha=0.6, label='Sampled points')
        ax1.plot(iterations, y_best_so_far, 'r-', linewidth=2, label='Best so far')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot parameters over iterations
        X_array = np.array(self.X_samples)
        for i in range(self.n_params):
            ax2.plot(iterations, X_array[:, i], 'o-', alpha=0.6, label=self.param_names[i])
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Parameter Value')
        ax2.set_title('Parameters Over Iterations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


# ============================================================================
# HELPER FUNCTIONS FOR COMMON OPTIMIZATION SCENARIOS
# ============================================================================

def optimize_yaw_angles(config: Config, yaw_bounds: List[Tuple[float, float]] = None, **opt_kwargs):
    """
    Optimize yaw angles for maximum power output.
    
    Parameters:
    -----------
    config : Config
        Base configuration
    yaw_bounds : List[Tuple[float, float]], optional
        Bounds for each turbine's yaw angle. If None, uses [-30, 30] for all turbines
    **opt_kwargs : Additional arguments passed to optimizer.optimize()
    
    Returns:
    --------
    BayesianOptimizer
        Optimizer with results
    """
    n_turbines = len(config.WindFarm)
    
    if yaw_bounds is None:
        bounds = np.array([[-30.0, 30.0]] * n_turbines)
    else:
        bounds = np.array(yaw_bounds)
    
    param_names = [f'yaw_T{i}' for i in range(n_turbines)]
    
    def yaw_mapping(params, cfg):
        cfg.WindFarm.yaw = params.copy()
        return cfg
    
    optimizer = Optimizer(
        config=config,
        bounds=bounds,
        param_mapping=yaw_mapping,
        param_names=param_names,
        **{k: v for k, v in opt_kwargs.items() if k not in ['n_iter', 'n_init', 'verbose']}
    )
    
    opt_params = {k: v for k, v in opt_kwargs.items() if k in ['n_iter', 'n_init', 'verbose']}
    best_params, best_value = optimizer.optimize(**opt_params)
    
    return optimizer, best_params, best_value


def optimize_turbine_positions(config: Config, x_bounds: List[Tuple[float, float]] = None, 
                                y_bounds: List[Tuple[float, float]] = None, **opt_kwargs):
    """
    Optimize turbine x,y positions for maximum power output.
    
    Parameters:
    -----------
    config : Config
        Base configuration
    x_bounds : List[Tuple[float, float]], optional
        Bounds for each turbine's x position
    y_bounds : List[Tuple[float, float]], optional
        Bounds for each turbine's y position
    **opt_kwargs : Additional arguments passed to optimizer.optimize()
    
    Returns:
    --------
    BayesianOptimizer
        Optimizer with results
    """
    n_turbines = len(config.WindFarm)
    
    if x_bounds is None:
        x_bounds = [(0.0, 1000.0)] * n_turbines
    if y_bounds is None:
        y_bounds = [(-500.0, 500.0)] * n_turbines
    
    # Interleave x and y bounds: [x0, y0, x1, y1, ...]
    bounds = []
    param_names = []
    for i in range(n_turbines):
        bounds.append(x_bounds[i])
        bounds.append(y_bounds[i])
        param_names.append(f'x_T{i}')
        param_names.append(f'y_T{i}')
    bounds = np.array(bounds)
    
    def position_mapping(params, cfg):
        # params = [x0, y0, x1, y1, ...]
        new_pos = cfg.WindFarm.pos.copy()
        for i in range(n_turbines):
            new_pos[i, 0] = params[2*i]     # x
            new_pos[i, 1] = params[2*i + 1] # y
        cfg.WindFarm.pos = new_pos
        return cfg
    
    optimizer = Optimizer(
        config=config,
        bounds=bounds,
        param_mapping=position_mapping,
        param_names=param_names,
        **{k: v for k, v in opt_kwargs.items() if k not in ['n_iter', 'n_init', 'verbose']}
    )
    
    opt_params = {k: v for k, v in opt_kwargs.items() if k in ['n_iter', 'n_init', 'verbose']}
    best_params, best_value = optimizer.optimize(**opt_params)
    
    return optimizer


def optimize_mixed_params(config: Config, param_specs: Dict[str, Any], **opt_kwargs):
    """
    Optimize any combination of parameters.
    
    Parameters:
    -----------
    config : Config
        Base configuration
    param_specs : Dict[str, Any]
        Dictionary specifying parameters to optimize. Format:
        {
            'param_name': {
                'bounds': (min, max),
                'indices': [0, 1, ...],  # For array parameters (e.g., which turbines)
                'path': 'WindFarm.yaw'   # Attribute path in config
            },
            ...
        }
    **opt_kwargs : Additional arguments passed to optimizer.optimize()
    
    Example:
    --------
    param_specs = {
        'yaw_T0': {'bounds': (-30, 30), 'indices': [0], 'path': 'WindFarm.yaw'},
        'yaw_T1': {'bounds': (-20, 20), 'indices': [1], 'path': 'WindFarm.yaw'},
        'Ct_T0': {'bounds': (0.5, 0.9), 'indices': [0], 'path': 'WindFarm.Ct'},
    }
    
    Returns:
    --------
    BayesianOptimizer
        Optimizer with results
    """
    param_names = list(param_specs.keys())
    bounds = np.array([spec['bounds'] for spec in param_specs.values()])
    
    def mixed_mapping(params, cfg):
        for i, (name, spec) in enumerate(param_specs.items()):
            path_parts = spec['path'].split('.')
            obj = cfg
            for part in path_parts[:-1]:
                obj = getattr(obj, part)
            
            attr_name = path_parts[-1]
            current_val = getattr(obj, attr_name)
            
            if isinstance(current_val, np.ndarray):
                new_val = current_val.copy()
                for idx in spec['indices']:
                    new_val[idx] = params[i]
                setattr(obj, attr_name, new_val)
            else:
                setattr(obj, attr_name, params[i])
        
        return cfg
    
    optimizer = Optimizer(
        config=config,
        bounds=bounds,
        param_mapping=mixed_mapping,
        param_names=param_names,
        **{k: v for k, v in opt_kwargs.items() if k not in ['n_iter', 'n_init', 'verbose']}
    )
    
    opt_params = {k: v for k, v in opt_kwargs.items() if k in ['n_iter', 'n_init', 'verbose']}
    best_params, best_value = optimizer.optimize(**opt_params)
    
    return optimizer, best_params, best_value


# ============================================================================
# TESTING
# ============================================================================

def test_yaw_optimization():
    """Test optimizing yaw angles."""
    config = Config()
    print("="*70)
    print("EXAMPLE 1: Optimizing Yaw Angles")
    print("="*70)
    
    optimizer = optimize_yaw_angles(
        config=config,
        yaw_bounds=[(-30, 30)] * len(config.WindFarm),
        acquisition_func='ei',
        xi=0.01,
        random_state=42,
        n_iter=10,
        n_init=3,
        verbose=True
    )
    
    optimizer.plot_optimization_history(save_path='yaw_optimization.png')
    return optimizer


def test_position_optimization():
    """Test optimizing turbine positions."""
    config = Config()
    print("\n" + "="*70)
    print("EXAMPLE 2: Optimizing Turbine Positions")
    print("="*70)
    
    n_turbines = len(config.WindFarm)
    optimizer = optimize_turbine_positions(
        config=config,
        x_bounds=[(0, 500)] * n_turbines,
        y_bounds=[(-200, 200)] * n_turbines,
        acquisition_func='ei',
        xi=0.01,
        random_state=42,
        n_iter=10,
        n_init=3,
        verbose=True
    )
    
    optimizer.plot_optimization_history(save_path='position_optimization.png')
    return optimizer


def test_mixed_optimization():
    """Test optimizing mixed parameters."""
    config = Config()
    print("\n" + "="*70)
    print("EXAMPLE 3: Optimizing Mixed Parameters (Yaw + Ct)")
    print("="*70)
    
    n_turbines = len(config.WindFarm)
    param_specs = {}
    
    # Add yaw for each turbine
    for i in range(n_turbines):
        param_specs[f'yaw_T{i}'] = {
            'bounds': (-30, 30),
            'indices': [i],
            'path': 'WindFarm.yaw'
        }
    
    # Add Ct for first turbine
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
        n_iter=10,
        n_init=3,
        verbose=True
    )
    
    optimizer.plot_optimization_history(save_path='mixed_optimization.png')
    return optimizer


if __name__ == "__main__":
    # Run examples
    opt1 = test_yaw_optimization()
    # opt2 = test_position_optimization()
    # opt3 = test_mixed_optimization()
