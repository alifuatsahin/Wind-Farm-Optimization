import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import gpytorch

from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.generation import MaxPosteriorSampling
from botorch.utils.sampling import SobolEngine
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
from copy import deepcopy

from simulation import Simulation
from config import Config

@dataclass
class State:
    """State used to track the recent history of the trust region."""
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        """Post-initialize the state of the trust region."""
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

class Optimizer:
    def __init__(
        self,
        config: Config,
        bounds: np.ndarray,
        param_mapping: Callable,
        param_names: Optional[List[str]] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64,
        random_state: Optional[int] = None,
        max_cholesky_size: Optional[int] = float("inf"),
    ):
        bounds = np.asarray(bounds, dtype=float)
        assert np.all(bounds[:, 1] > bounds[:, 0]), "Error: Upper bounds must be greater than lower bounds"
        assert max_cholesky_size >= 0, "Error: max_cholesky_size must be non-negative"
        assert dtype in [torch.float32, torch.float64], "Error: dtype must be torch.float32 or torch.float64"
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA device requested but not available. Falling back to CPU.")
            device = "cpu"

        self.config = config
        self.bounds = bounds
        self.n_params = len(bounds)
        self.param_mapping = param_mapping
        self.param_names = param_names or [f'param_{i}' for i in range(self.n_params)]
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_cholesky_size = max_cholesky_size
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        self.X = torch.empty((0, self.n_params), dtype=self.dtype, device=self.device)
        self.Y = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        self._X = torch.empty((0, self.n_params), dtype=self.dtype, device=self.device)
        self._Y = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        self.X_opt = []
        self.Y_opt = []
        self.best_value = -np.inf
        self.best_params = None

    def _restart(self):
        """Reset the optimizer state for a new trust region restart."""
        self._X = torch.empty((0, self.n_params), dtype=self.dtype, device=self.device)
        self._Y = torch.empty((0, 1), dtype=self.dtype, device=self.device)

    def _normalize(self, x_physical: np.ndarray) -> np.ndarray:
        """Normalize parameters from physical bounds to [0, 1]."""
        return (x_physical - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def _denormalize(self, x_normalized: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [0, 1] to physical bounds."""
        return x_normalized * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
    
    def _evaluate_objective(self, params: np.ndarray) -> float:
        """Evaluates wind farm power for given normalized parameters."""
        physical_params = self._denormalize(params)
        temp_config = self.param_mapping(physical_params, deepcopy(self.config))
        
        sim = Simulation(temp_config)
        sim.run()
        return sim.calculate_objective()

    def _get_initial_points(self, n_samples: int) -> torch.Tensor:
        """Generate initial points using Sobol sequence."""
        sobol = SobolEngine(dimension=self.n_params, scramble=True)
        return sobol.draw(n_samples).to(dtype=self.dtype, device=self.device)

    def _update_state(self, state: 'State', Y_next: torch.Tensor) -> 'State':
        """Update the state of the trust region based on the new function values."""
        if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1

        if state.success_counter == state.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        state.best_value = max(state.best_value, max(Y_next).item())
        if state.length < state.length_min:
            state.restart_triggered = True
        return state
    
    def _compute_tr_bounds(self, model: SingleTaskGP, X: torch.Tensor, Y: torch.Tensor, state: 'State') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the trust region bounds."""
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
        return tr_lb, tr_ub, x_center
    
    def _sample_candidates(self, x_center: torch.Tensor, tr_lb: torch.Tensor, tr_ub: torch.Tensor, n_candidates: int) -> torch.Tensor:
        """Sample candidate points within the trust region."""
        dim = tr_lb.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=self.dtype, device=self.device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        return X_cand

    def _generate_batch(
        self,
        state: 'State',
        acqf_type: str,
        model: SingleTaskGP,  # GP model
        X: torch.Tensor,  # Evaluated points on the domain [0, 1]^d
        Y: torch.Tensor,  # Function values
        batch_size: int,
        num_restarts: int = 10, # Number of restarts for qEI optimization
        raw_samples: int = 512, # Number of raw samples for qEI optimization
        n_candidates: Optional[int] = None,  # Number of candidates for Thompson sampling
    ) -> torch.Tensor:
        """Generate a new batch of points."""
        assert acqf_type in ['ts', 'ei'], "acqf_type must be 'ts' or 'ei'"
        assert X.min() >= 0.0
        assert X.max() <= 1.0
        assert torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Compute trust region bounds
        tr_lb, tr_ub, x_center = self._compute_tr_bounds(model, X, Y, state)

        if acqf_type == 'ts':
            # Sample candidate points within the trust region
            X_cand = self._sample_candidates(x_center, tr_lb, tr_ub, n_candidates)

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = thompson_sampling(X_cand, num_samples=batch_size)
        
        elif acqf_type == 'ei':
            ei = qExpectedImprovement(model=model, best_f=Y.max())
            X_next, _ = optimize_acqf(
                acq_function=ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        return X_next
    
    def _fit_gp(self, 
            X: torch.Tensor, 
            Y: torch.Tensor, 
            dim: int, 
        ) -> SingleTaskGP:
        
        """Fit a GP model to the data."""
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-2))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 2.0))
        )
        model = SingleTaskGP(X, Y, covar_module=covar_module, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            fit_gpytorch_mll(mll)
        
        return model

    def optimize(self, n_init: int = 20, 
                 max_evals: int = 100,
                 batch_size: int = 1, 
                 n_candidates: int = 5000,
                 acquisition_func: str = 'ts',
                 log_freq: int = 5,
                 verbose: bool = True
            ) -> Tuple[np.ndarray, float]:
        
        """
        Run the Bayesian Optimization loop with trust region and restarts.
        
        Parameters
        ----------
        n_init : int
            Number of initial random samples (only for first run)
        max_evals : int
            Maximum number of function evaluations
        batch_size : int
            Number of points to evaluate in parallel
        n_candidates : int
            Number of candidate points for Thompson Sampling
        acquisition_func : str
            Acquisition function type ('ts' or 'ei')
        verbose : bool
            Print progress information
            
        Returns
        -------
        best_params : np.ndarray
            Best parameters found (in physical space)
        best_value : float
            Best objective value achieved
        """
        if verbose:
            print(f"Starting optimization on device: {self.device}")
            print(f"Max evaluations: {max_evals}\n")
            # Initial Sobol sampling (only once at the beginning)
            print(f"=== Initial Sobol Sampling ({n_init} points) ===")

        n_evals = 0
        
        # 2. Restart loop
        while n_evals < max_evals:
            # Initialize data storage for this TR
            self._restart()

            # Generate and evaluate initial points
            n_init_curr = min(n_init, max_evals - n_evals)
            init_X = self._get_initial_points(n_init_curr)
            init_Y = torch.tensor(
                [self._evaluate_objective(x.cpu().numpy()) for x in init_X], 
                dtype=self.dtype, device=self.device
            ).unsqueeze(-1)

            # Initialize trust region state
            state = State(
                dim=self.n_params, 
                batch_size=batch_size, 
                best_value=init_Y.max().item()
            )

            # Convert to local data
            n_evals += n_init_curr
            self._X = init_X.clone()
            self._Y = init_Y.clone()

            # Update global data
            self.X = torch.cat([self.X, init_X.clone()], dim=0)
            self.Y = torch.cat([self.Y, init_Y.clone()], dim=0)
            
            if init_Y.max() > self.best_value:
                # Initialize best
                best_idx = self._Y.argmax()
                self.best_value = self._Y[best_idx].item()
                self.best_params = self._denormalize(self._X[best_idx].cpu().numpy())
            
            # Main optimization loop
            while n_evals < max_evals and not state.restart_triggered:
                # Standardize targets for GP fitting
                train_Y = (self._Y - self._Y.mean()) / self._Y.std()
                
                # Fit GP model on all accumulated data
                gp = self._fit_gp(self._X, train_Y, dim=self.n_params)
                
                # Generate candidates within trust region
                with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                    x_next = self._generate_batch(
                        state=state,
                        model=gp,
                        acqf_type=acquisition_func,
                        X=self._X,
                        Y=train_Y.squeeze(-1),
                        batch_size=batch_size,
                        n_candidates=n_candidates,
                    )
                
                # Evaluate candidates
                y_next = torch.tensor(
                    [self._evaluate_objective(x.cpu().numpy()) for x in x_next],
                    dtype=self.dtype, device=self.device
                ).unsqueeze(-1)

                # Update global data
                self._X = torch.cat([self._X, x_next], dim=0)
                self._Y = torch.cat([self._Y, y_next], dim=0)
                n_evals += batch_size

                # Update trust region state
                state = self._update_state(state, y_next.squeeze(-1))

                # Update global best
                if self._Y.max() > self.best_value:
                    best_idx = self._Y.argmax()
                    self.best_value = self._Y[best_idx].item()
                    self.best_params = self._denormalize(self._X[best_idx].cpu().numpy())
                    if verbose:
                        print(f"Eval {n_evals}/{max_evals}: New best found! {self.best_value:.4f} (TR: {state.length:.3f})")
                elif verbose and n_evals % log_freq == 0:
                    print(f"Eval {n_evals}/{max_evals}: Best={self.best_value:.4f}, TR length={state.length:.3f}")

                self.X = torch.cat([self.X, x_next.clone()], dim=0)
                self.Y = torch.cat([self.Y, y_next.clone()], dim=0)

        # Store final samples
        self.X_opt = self.X.cpu().numpy().tolist()
        self.Y_opt = self.Y.squeeze(-1).cpu().numpy().tolist()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimization complete!")
            print(f"Total evaluations: {n_evals}/{max_evals}")
            print(f"Best value found: {self.best_value:.4f}")
            print(f"{'='*60}")

        return self.best_params, self.best_value
    
    def save_results(self, filepath: str):
        """
        Save optimization results to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the results
        """
        import pickle
        results = {
            'X_samples': self.X_opt,
            'y_samples': self.Y_opt,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'bounds': self.bounds.tolist(),
            'param_names': self.param_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Optimization results saved to {filepath}")
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot the optimization history.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """        
        iterations = np.arange(len(self.Y_opt))
        y_best_so_far = np.maximum.accumulate(self.Y_opt)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot objective value over iterations
        ax1.plot(iterations, self.Y_opt, 'bo-', alpha=0.6, label='Sampled points')
        ax1.plot(iterations, y_best_so_far, 'r-', linewidth=2, label='Best so far')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot parameters over iterations
        X_array = np.array(self.X_opt)
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
    
    def plot_trust_region_snapshots(self, iterations: List[int], param_idx_x: int = 0, 
                                    param_idx_y: int = 1, 
                                    save_path: Optional[str] = None):
        """
        Plot snapshots of trust region evolution at specified iterations.
        Shows how the trust region adapts during optimization.
        
        Parameters:
        -----------
        iterations : List[int]
            Iteration numbers to plot (e.g., [5, 15, 25, 35])
        param_idx_x : int
            Parameter index for x-axis (default: 0)
        param_idx_y : int
            Parameter index for y-axis (default: 1)
        save_path : str, optional
            Path to save the figure
        """
        if self.n_params < 2:
            print("Warning: Need at least 2 parameters for 2D plot")
            return
        
        n_plots = len(iterations)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        for idx, iter_num in enumerate(iterations):
            if iter_num >= len(self.X_opt):
                print(f"Warning: Iteration {iter_num} not available (only {len(self.X_opt)} samples)")
                continue
            
            # Get data up to this iteration
            X_train = np.array(self.X_opt[:iter_num+1])
            y_train = np.array(self.Y_opt[:iter_num+1])
            
            # Plot
            ax = axes[idx]
            
            # Plot all sampled points
            scatter = ax.scatter(X_train[:, param_idx_x], X_train[:, param_idx_y], 
                               c=y_train, cmap='viridis', s=100, edgecolors='k', 
                               alpha=0.7, label='Sampled')
            
            # Highlight best point
            best_idx = np.argmax(y_train)
            ax.scatter([X_train[best_idx, param_idx_x]], [X_train[best_idx, param_idx_y]], 
                      c='red', marker='*', s=400, edgecolors='k', 
                      linewidths=2, label='Best', zorder=5)
            
            # Highlight most recent point
            ax.scatter([X_train[-1, param_idx_x]], [X_train[-1, param_idx_y]], 
                      c='cyan', marker='o', s=200, edgecolors='k', 
                      linewidths=2, label='Latest', zorder=5)
            
            ax.set_xlabel(self.param_names[param_idx_x], fontsize=10)
            ax.set_ylabel(self.param_names[param_idx_y], fontsize=10)
            ax.set_title(f'Iteration {iter_num}\nBest={np.max(y_train):.3f}', fontsize=11)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.colorbar(scatter, ax=ax, label='Objective Value')
            if idx == 0:
                ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trust region snapshots saved to {save_path}")
        else:
            plt.show()
    
    def plot_acquisition_function(self, iterations: List[int], param_idx_x: int = 0,
                                 param_idx_y: int = 1, n_points: int = 50,
                                 acqf_type: str = 'ei',
                                 save_path: Optional[str] = None):
        """
        Plot the acquisition function surface at specified iterations.
        Only works with 'ei' acquisition function (cannot visualize Thompson Sampling).
        
        Parameters:
        -----------
        iterations : List[int]
            Iteration numbers to plot
        param_idx_x : int
            Parameter index for x-axis
        param_idx_y : int
            Parameter index for y-axis
        n_points : int
            Grid resolution
        acqf_type : str
            Acquisition function type ('ei' only for visualization)
        save_path : str, optional
            Path to save the figure
        """

        assert acqf_type in ['ei'], "Error: Acquisition function visualization only supports 'ei' (Expected Improvement)"
        assert self.n_params >= 2, "Error: Need at least 2 parameters for 2D acquisition function plot"
        
        n_plots = len(iterations)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        # Create meshgrid
        x_vals = np.linspace(0, 1, n_points)
        y_vals = np.linspace(0, 1, n_points)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        
        for idx, iter_num in enumerate(iterations):
            if iter_num >= len(self.X_opt):
                print(f"Warning: Iteration {iter_num} not available")
                continue
            
            # Fit GP up to this iteration
            X_train = torch.tensor(self.X_opt[:iter_num+1], dtype=self.dtype, device=self.device)
            y_train = torch.tensor(self.Y_opt[:iter_num+1], dtype=self.dtype, device=self.device).unsqueeze(-1)
            
            # Fit model
            gp = self._fit_gp(X_train, y_train, dim=self.n_params)
            gp.eval()
            
            # Compute acquisition on grid
            Z_acq = np.zeros_like(X_grid)
            best_params = X_train[y_train.argmax()].cpu().numpy()
            best_f = y_train.max().item()
            
            # Create EI acquisition function
            ei = qExpectedImprovement(model=gp, best_f=best_f)
            
            for i in range(n_points):
                for j in range(n_points):
                    x_test = best_params.copy()
                    x_test[param_idx_x] = X_grid[i, j]
                    x_test[param_idx_y] = Y_grid[i, j]
                    x_test_t = torch.tensor(x_test, dtype=self.dtype, device=self.device).unsqueeze(0)
                    
                    with torch.no_grad():
                        acq_value = ei(x_test_t.unsqueeze(0))  # Add batch dimension
                        Z_acq[i, j] = acq_value.item()
            
            # Plot
            ax = axes[idx]
            c = ax.contourf(X_grid, Y_grid, Z_acq, levels=20, cmap='viridis')
            
            # Overlay sampled points
            X_train_np = X_train.cpu().numpy()
            ax.scatter(X_train_np[:, param_idx_x], X_train_np[:, param_idx_y],
                      c='red', marker='x', s=80, alpha=0.7, label='Sampled', linewidths=2)
            
            # Highlight next point if available
            if iter_num < len(self.X_opt) - 1:
                next_x = self.X_opt[iter_num + 1]
                ax.scatter([next_x[param_idx_x]], [next_x[param_idx_y]],
                          c='cyan', marker='*', s=300, edgecolors='k',
                          linewidths=2, label='Next', zorder=5)
            
            ax.set_xlabel(self.param_names[param_idx_x], fontsize=10)
            ax.set_ylabel(self.param_names[param_idx_y], fontsize=10)
            ax.set_title(f'Iteration {iter_num}\nMax EI={np.max(Z_acq):.3e}', fontsize=11)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.colorbar(c, ax=ax, label='Expected Improvement')
            if idx == 0:
                ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Acquisition function plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_gp_posterior_snapshots(self, iterations: List[int], param_idx_x: int = 0, 
                                   param_idx_y: int = 1, n_points: int = 50, 
                                   save_path: Optional[str] = None):
        """
        Plot snapshots of GP posterior mean and uncertainty at specified iterations.
        Shows how the surrogate model learns the objective landscape.
        
        Parameters:
        -----------
        iterations : List[int]
            Iteration numbers to plot (e.g., [10, 20, 30, 40])
        param_idx_x : int
            Parameter index for x-axis
        param_idx_y : int
            Parameter index for y-axis
        n_points : int
            Grid resolution
        save_path : str, optional
            Path to save the figure
        """
        if self.n_params < 2:
            print("Warning: Need at least 2 parameters for 2D posterior plot")
            return
        
        n_plots = len(iterations)
        fig, axes = plt.subplots(2, n_plots, figsize=(5*n_plots, 8))
        if n_plots == 1:
            axes = axes.reshape(2, 1)
        
        # Create meshgrid
        x_vals = np.linspace(0, 1, n_points)
        y_vals = np.linspace(0, 1, n_points)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        
        for idx, iter_num in enumerate(iterations):
            if iter_num >= len(self.X_opt):
                print(f"Warning: Iteration {iter_num} not available")
                continue
            
            # Fit GP up to this iteration
            X_train = torch.tensor(self.X_opt[:iter_num+1], dtype=self.dtype, device=self.device)
            y_train = torch.tensor(self.Y_opt[:iter_num+1], dtype=self.dtype, device=self.device).unsqueeze(-1)
            
            # Fit model
            gp = self._fit_gp(X_train, y_train, dim=self.n_params)
            gp.eval()
            
            # Compute posterior on grid
            Z_mean = np.zeros_like(X_grid)
            Z_std = np.zeros_like(X_grid)
            
            best_params = X_train[y_train.argmax()].cpu().numpy()
            
            for i in range(n_points):
                for j in range(n_points):
                    x_test = best_params.copy()
                    x_test[param_idx_x] = X_grid[i, j]
                    x_test[param_idx_y] = Y_grid[i, j]
                    x_test_t = torch.tensor(x_test, dtype=self.dtype, device=self.device).unsqueeze(0)
                    
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        pred = gp.posterior(x_test_t)
                        Z_mean[i, j] = pred.mean.cpu().numpy()[0, 0]
                        Z_std[i, j] = pred.variance.sqrt().cpu().numpy()[0, 0]
            
            # Plot mean
            ax_mean = axes[0, idx]
            c1 = ax_mean.contourf(X_grid, Y_grid, Z_mean, levels=20, cmap='coolwarm')
            X_train_np = X_train.cpu().numpy()
            y_train_np = y_train.squeeze(-1).cpu().numpy()
            ax_mean.scatter(X_train_np[:, param_idx_x], X_train_np[:, param_idx_y], 
                           c=y_train_np, cmap='coolwarm', edgecolors='k', 
                           s=50, marker='o', label='Sampled', zorder=5)
            ax_mean.set_xlabel(self.param_names[param_idx_x], fontsize=10)
            ax_mean.set_ylabel(self.param_names[param_idx_y], fontsize=10)
            ax_mean.set_title(f'Iter {iter_num}: Posterior Mean', fontsize=11)
            ax_mean.set_xlim(0, 1)
            ax_mean.set_ylim(0, 1)
            plt.colorbar(c1, ax=ax_mean, label='Predicted Value')
            
            # Plot uncertainty (std)
            ax_std = axes[1, idx]
            c2 = ax_std.contourf(X_grid, Y_grid, Z_std, levels=20, cmap='plasma')
            ax_std.scatter(X_train_np[:, param_idx_x], X_train_np[:, param_idx_y], 
                          c='white', edgecolors='k', s=50, marker='o', 
                          alpha=0.7, zorder=5)
            ax_std.set_xlabel(self.param_names[param_idx_x], fontsize=10)
            ax_std.set_ylabel(self.param_names[param_idx_y], fontsize=10)
            ax_std.set_title(f'Iter {iter_num}: Uncertainty (σ)', fontsize=11)
            ax_std.set_xlim(0, 1)
            ax_std.set_ylim(0, 1)
            plt.colorbar(c2, ax=ax_std, label='Std Dev')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GP posterior snapshots saved to {save_path}")
        else:
            plt.show()