from .superposition import get_local_velocity_field
from .turbine_state import make_turbine_params, LocalConditions
from . import turbine_physics as tp

import numpy as np
import pandas as pd
import os

class Turbine:
    def __init__(self, config, field_params):
        self.config = config
        self.field_params = field_params

        self._params = make_turbine_params(config, field_params)
        local = tp.init_local_conditions(self._params)

        self.pos = self._params.pos  # (x, y, z)
        self.D = self._params.D
        self.Zhub = self._params.Zhub
        self.yaw = self._params.yaw
        self.TSR = self._params.TSR
        self.Uh = self._params.Uh
        self.Zh = self._params.Zh
        self.WV = self._params.WV
        self.Nv = self._params.Nv
        self.vortex_field = None  # to be filled after simulation
        self.wake_field = None  # to be filled after wake calculation
        self.dl = None  # grid spacing in y direction
        self.calculation_domain = 0.0  # to be set by WindFarm

        self.Ct = self._params.Ct
        self.Cp = self._params.Cp  # Adjusted power coefficient

        self.phi = self._params.phi
        self.dphi = self._params.dphi
        self.yloc = self._params.yloc
        self.zloc = self._params.zloc

        self.V = local.V
        self.W = local.W
        self.Uin = local.Uin
        self.Uhub = local.Uhub

    def _current_local(self):
        """Snapshot of the state WindFarm.solve() mutates directly on this instance."""
        return LocalConditions(Uhub=self.Uhub, V=self.V, W=self.W, Uin=self.Uin)

    def init_Uin(self):
        return tp.init_Uin(self._params)

    @property
    def a(self):
        return self._params.a

    @property
    def R(self):
        return self._params.R

    @property
    def Rv(self):
        return self._params.Rv

    @property
    def omega(self):
        return tp.compute_omega(self._params, self.Uhub)

    @property
    def beta(self):
        return self._params.beta

    @property
    def Yoffset(self):
        return self._params.Yoffset

    @property
    def gamma0(self):
        return tp.compute_gamma0(self._params, self.Uhub)

    @property
    def Ut(self):
        return tp.compute_Ut(self._params, self.Uhub)

    @property
    def Uinf(self):
        return tp.nominal_hub_velocity(self._params)

    @property
    def dgamma(self):
        return tp.compute_dgamma(self._params, self.Uhub)

    def calculate_efficiency(self):
        return tp.calculate_efficiency(self._params, self.Uhub)

    def simulate_vortex_field(self):
        self._params.calculation_domain = self.calculation_domain
        self.vortex_field = tp.simulate_vortex_field(self._params, self._current_local())

    def initialize_wake_field(self):
        self.vortex_field, self.dl = tp.initialize_wake_field(self._params, self.vortex_field, self._current_local())

    def calculate_deficit_field(self, upstream_turbines, max_steps=1500, N_upstream_max=None):
        self._params.calculation_domain = self.calculation_domain
        self.wake_field = tp.calculate_deficit_field(
            self._params, self._current_local(), self.vortex_field, self.dl, upstream_turbines,
            N_upstream_max=N_upstream_max, max_steps=max_steps
        )

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
        # Farm-wide constant so NuT_model's upstream-turbine padding (see
        # turbine_state.pack_upstream_turbines) is the SAME shape for every turbine's call --
        # required for calculate_deficit_field's jit to compile once and serve every turbine,
        # not recompile per turbine's distinct (varying-by-construction) upstream count.
        N_upstream_max = max(len(self.turbines) - 1, 0)
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
            t.calculate_deficit_field(upstream_turbines, N_upstream_max=N_upstream_max)

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