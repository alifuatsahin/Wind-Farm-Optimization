from .utils import smooth_2d, NuT_model
from .vortex_model import simulate_vortex_evolution
from .model_solver import advance_wake_field
from .superposition import get_local_velocity_field, interpolate_vec_data

import numpy as np
import pandas as pd
import os

class Turbine:
    def __init__(self, config, field_params):
        self.config = config
        self.field_params = field_params
        self.pos = config.pos  # (x, y, z)
        self.D = config.D
        self.Zhub = config.Zhub
        self.yaw = config.yaw
        self.TSR = config.TSR
        self.Uh = field_params.Uh
        self.Zh = field_params.Zh
        self.WV = field_params.WV
        self.Nv = field_params.Nv
        self.vortex_field = None  # to be filled after simulation
        self.wake_field = None  # to be filled after wake calculation
        self.dl = None  # grid spacing in y direction
        self.calculation_domain = 0.0  # to be set by WindFarm

        self.Ct = config.Ct
        self.Cp = config.Cp * np.cos(self.beta)**1.88  # Adjusted power coefficient

        self.phi = np.linspace(-np.pi, np.pi, self.Nv)
        self.dphi = abs(self.phi[1] - self.phi[0])
        self._initialize_grid()

        self.V = np.zeros_like(self.yloc)
        self.W = np.zeros_like(self.zloc)

        self.Uin = self.init_Uin()
        self.Uhub = self._init_Uhub()

    def _initialize_grid(self):
        Ly = self.field_params.max_Y * self.D
        Lz = self.field_params.max_Z * self.D
        n_grids = self.field_params.n_grids

        Ny = max(2, int(Ly / (self.D / n_grids)))
        Nz = max(2, int(Lz / (self.D / n_grids)))
        if self.Zhub - Lz/2 < 0:
            zlims = (0, Lz)
        else:
            zlims = (self.Zhub - Lz/2, self.Zhub + Lz/2)
        self.yloc, self.zloc = np.meshgrid(np.linspace(-Ly/2, Ly/2, Ny), np.linspace(*zlims, Nz), indexing='ij')

    def _compute_Ut(self):
        z = self.Zhub + self.D / 2.0 * np.sin(self.phi)
        zsafe = np.maximum(z, self.field_params.z0 + 1e-6)
        Utbl = self.Uh * (np.log(zsafe / self.field_params.z0) / np.log(self.Zh / self.field_params.z0))
        U_tx = (1 - self.a) * Utbl - self.omega * self.R * np.sin(self.phi) * np.sin(self.beta)
        U_ty = -self.omega * self.R * np.sin(self.phi) * np.cos(self.beta)
        U_tz = self.omega * self.R * np.cos(self.phi)
        return np.array([U_tx, U_ty, U_tz]).T

    def _compute_dgamma(self):
        alpha = np.arcsin(self.Ut[:, 0] / np.sqrt(np.sum(self.Ut ** 2, axis=1)))  # Fixed indexing
        dgamma = np.sin(alpha) * self.dphi
        gamma_ref = self.gamma0 * 0.45
        dgamma = gamma_ref / np.sum(dgamma[1:]) * dgamma
        return dgamma

    def _init_Uhub(self):
        Uin = self.init_Uin()
        rotor_mask = np.sqrt((self.yloc)**2 + (self.zloc - self.Zhub)**2) <= (self.D / 2)
        Uhub = np.mean(Uin[rotor_mask])
        return Uhub

    def init_Uin(self):
        zsafe = np.maximum(self.zloc, self.field_params.z0 + 1e-6)  # avoid log(0) issues
        Uin = self.Uh * (np.log(zsafe / self.field_params.z0) / np.log(self.Zh / self.field_params.z0))
        return Uin

    @property
    def a(self):
        return (1 - np.sqrt(1 - self.Ct / np.cos(self.beta))) / 2

    @property
    def R(self):
        return self.D / 2

    @property
    def Rv(self):
        return 0.1 * self.D

    @property
    def omega(self):
        return self.TSR * self.Uhub / self.R

    @property
    def beta(self):
        return np.deg2rad(self.yaw)

    @property
    def Yoffset(self):
        return 5 * np.sin(self.beta)

    @property
    def gamma0(self):
        return np.pi * self.Uhub ** 2 * self.Ct / self.omega

    @property
    def Ut(self):
        return self._compute_Ut()
    
    @property
    def Uinf(self):
        return self._init_Uhub()

    @property
    def dgamma(self):
        return self._compute_dgamma()        
    
    def calculate_efficiency(self):
        P = (self.Uhub ** 3) * self.Cp
        nominal_P = (self.Uinf ** 3) * self.config.Cp
        return P / nominal_P

    def simulate_vortex_field(self):
        self.vortex_field = simulate_vortex_evolution(self, self.field_params)
    
    def initialize_wake_field(self):
        yloc = self.vortex_field[0].yloc
        zloc = self.vortex_field[0].zloc
        beta = self.beta
        
        self.dl = float(yloc[1, 0] - yloc[0, 0])
        U = self.Uin.copy()
        mask = np.sqrt(((yloc + self.Yoffset)**2) / (np.cos(beta)**2) + (zloc - self.Zhub)**2) <= self.R
        U[mask] -= 2.0 * U[mask] * self.a

        hub_mask = (np.abs(yloc) <= self.dl * 1.0) & (zloc < self.Zhub)
        U[hub_mask] -= 0.3 * U[hub_mask]  # add some velocity deficit at the hub

        U_smooth = smooth_2d(U, kernel_size=3)
        self.vortex_field[0].U = U_smooth
        self.vortex_field[0].X = 0.0
        self.vortex_field[0].t = 0.0
        self.vortex_field[0].Uhub = self.Uhub

    def calculate_deficit_field(self, upstream_turbines, max_steps=10000):
        # Time-marching reduced-order model
        self.wake_field = [self.vortex_field[0]]

        while self.wake_field[-1].X <= self.calculation_domain:
            NuT = NuT_model(self.wake_field[-1], self, self.field_params, upstream_turbines)
            dt = min(self.dl / self.Uhub, 0.25 * self.D / self.Uhub) 
            dt = min(dt, (self.dl**2) / (2 * (NuT + 1e-6)))  # stability condition

            new = interpolate_vec_data(self.vortex_field, self.wake_field[-1].t + dt)
            U, X_new = advance_wake_field(self.wake_field[-1], dt, NuT, self, self.field_params)
            new.U = U
            new.X = X_new
            new.t = self.wake_field[-1].t + dt
            self.wake_field.append(new)

            if len(self.wake_field) > max_steps:
                print("Safety break: Too many steps, stopping simulation.")
                break
    
class WindFarm:
    def __init__(self, config):
        self.turbine_configs = config.WindFarm
        self.field_params = config.Field
        self.turbines = []

        self._construct_wind_farm()

    def get_grid(self, tolerance=None):
        """
        Return grid dimensions and layout of the wind farm.
        Handles misaligned turbines by clustering positions within a tolerance.
        
        Args:
            tolerance: Distance tolerance for grouping positions (default: D/3)
        
        Returns:
            dict: Contains 'rows', 'cols', 'x_positions', 'y_positions', 'layout'
        """
        if not self.turbines:
            return {'rows': 0, 'cols': 0, 'x_positions': [], 'y_positions': [], 'layout': None}
        
        # Use rotor diameter as default tolerance if not specified
        if tolerance is None:
            tolerance = self.turbines[0].D / 3.0
        
        # Cluster x positions (streamwise)
        x_coords = [t.pos[0] for t in self.turbines]
        x_positions = self._cluster_positions(x_coords, tolerance)
        
        # Cluster y positions (spanwise)
        y_coords = [t.pos[1] for t in self.turbines]
        y_positions = self._cluster_positions(y_coords, tolerance)
        
        rows = len(x_positions)
        cols = len(y_positions)
        
        # Create a 2D layout grid
        layout = [[None for _ in range(cols)] for _ in range(rows)]
        
        for idx, turbine in enumerate(self.turbines):
            # Find closest cluster centers
            row_idx = self._find_closest_cluster(turbine.pos[0], x_positions)
            col_idx = self._find_closest_cluster(turbine.pos[1], y_positions)
            layout[row_idx][col_idx] = idx
        
        return {
            'rows': rows,
            'cols': cols,
            'x_positions': x_positions,
            'y_positions': y_positions,
            'layout': np.array(layout, dtype=object),
            'total_turbines': len(self.turbines)
        }

    def _cluster_positions(self, positions, tolerance):
        """
        Cluster positions that are within tolerance distance.
        Returns sorted list of cluster centers.
        """
        if not positions:
            return []
        
        sorted_pos = sorted(positions)
        clusters = []
        current_cluster = [sorted_pos[0]]
        
        for pos in sorted_pos[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                # Save cluster center (mean of positions in cluster)
                clusters.append(np.mean(current_cluster))
                current_cluster = [pos]
        
        # Don't forget the last cluster
        clusters.append(np.mean(current_cluster))
        
        return clusters

    def _find_closest_cluster(self, position, cluster_centers):
        """
        Find index of closest cluster center to given position.
        """
        distances = [abs(position - center) for center in cluster_centers]
        return distances.index(min(distances))
    
    def _get_calculation_domain(self):
        """
        Calculate the downstream domain for each turbine as the distance to the furthest downstream turbine,
        with an additional buffer proportional to the rotor diameter.
        """
        if not self.turbines:
            return
        x_positions = [t.pos[0] for t in self.turbines]
        max_x = max(x_positions)
        buffer = self.field_params.min_X * self.turbines[0].D
        for t in self.turbines:
            t.calculation_domain = (max_x + buffer - t.pos[0])

    def _construct_wind_farm(self):
        self.turbines = [Turbine(t_config, self.field_params) for t_config in self.turbine_configs.turbines()]
        self.turbines = sorted(self.turbines, key=lambda t: t.pos[0])
        self._get_calculation_domain()

    def calculate_efficiency(self, verbose=False):
        total_eff = 0.0
        for t in self.turbines:
            eta = t.calculate_efficiency()
            if verbose:
                print(f"Turbine at pos={t.pos} m, yaw={t.yaw}°: Efficiency = {eta * 100:.2f} %")
            total_eff += eta
        total_eff /= len(self.turbines)
        if verbose:
            print(f"Wind Farm Average Efficiency: {total_eff * 100:.2f} %")
        return total_eff

    def solve(self):
        for t in self.turbines:
            U_local, V_local, W_local = get_local_velocity_field(t, self, method='MCS')

            upstream_turbines = [
                ut for ut in self.turbines
                if (
                    ut.pos[0] < t.pos[0] and  # Check if upstream
                    np.abs(ut.pos[1] - t.pos[1]) < 3 * ut.D and  # Check lateral distance
                    np.abs(ut.pos[2] - t.pos[2]) < 3 * ut.D  # Check vertical distance
                )
            ]

            # Create a mask for the rotor disk
            R = t.D / 2.0
            dist_from_hub = np.sqrt((t.yloc / np.cos(t.beta))**2 + (t.zloc - t.Zhub)**2)
            rotor_mask = dist_from_hub <= R
            
            t.Uhub = np.mean(U_local[rotor_mask])
            t.V = V_local
            t.W = W_local
            t.Uin = U_local

            # print(f"Simulating turbine at pos={t.pos} m, yaw={t.yaw}°")
            t.simulate_vortex_field()
            t.initialize_wake_field()
            t.calculate_deficit_field(upstream_turbines)

    def save_results(self, out_path, limit_frames=None):
        os.makedirs(out_path, exist_ok=True)
        for i, t in enumerate(self.turbines):
            rows = []
            data = t.wake_field[::max(1, len(t.wake_field)//limit_frames)] if limit_frames is not None else t.wake_field
            for fi, frame in enumerate(data):
                yloc = getattr(frame, "yloc", None)
                zloc = getattr(frame, "zloc", None)
                if yloc is None or zloc is None:
                    continue
                U = getattr(frame, "U", None)
                V = getattr(frame, "V", None)
                W = getattr(frame, "W", None)
                OmegaX = getattr(frame, "OmegaX", None)

                Ny, Nz = yloc.shape
                y_flat = yloc.ravel()
                z_flat = zloc.ravel()
                U_flat = U.ravel() if (U is not None) else np.full(Ny * Nz, np.nan)
                V_flat = V.ravel() if (V is not None) else np.full(Ny * Nz, np.nan)
                W_flat = W.ravel() if (W is not None) else np.full(Ny * Nz, np.nan)
                Om_flat = OmegaX.ravel() if (OmegaX is not None) else np.full(Ny * Nz, np.nan)

                Xval = float(getattr(frame, "X", np.nan))
                tval = float(getattr(frame, "t", np.nan))

                # build rows efficiently as list of dicts
                for idx in range(Ny * Nz):
                    rows.append({
                        "turbine_index": i,
                        "yaw": float(t.yaw),
                        "frame_index": fi,
                        "X": Xval,
                        "t": tval,
                        "y_idx": int(idx // Nz),
                        "z_idx": int(idx % Nz),
                        "y": float(y_flat[idx]),
                        "z": float(z_flat[idx]),
                        "U": float(U_flat[idx]),
                        "V": float(V_flat[idx]),
                        "W": float(W_flat[idx]),
                        "OmegaX": float(Om_flat[idx]),
                    })

            if not rows:
                continue

            df = pd.DataFrame(rows)
            base = os.path.join(out_path, f"Turbine_{i}_Yaw{t.yaw:.2f}")
            csv_path = base + ".csv"
            df.to_csv(csv_path, index=False)