from .wake_solver import solve_single_field
from .turbine_state import make_turbine_params, LocalConditions
from . import turbine_physics as tp

import numpy as np
import pandas as pd
import os

class Turbine:
    def __init__(self, config, field_params, y_bounds=None, n_rotors=1):
        self.config = config
        self.field_params = field_params

        self._params = make_turbine_params(config, field_params, y_bounds, n_rotors)
        local = tp.init_local_conditions(self._params)

        self.vortex_field = None
        self.vortex_exhausted = False
        self.wake_field = None
        self.dl = None
        self.calculation_domain = 0.0

        self.V = local.V
        self.W = local.W
        self.Uin = local.Uin
        self.Uhub = local.Uhub
        self.Uhub3 = local.Uhub

    pos = property(lambda self: self._params.pos)
    D = property(lambda self: self._params.D)
    Zhub = property(lambda self: self._params.Zhub)
    yaw = property(lambda self: self._params.yaw)
    TSR = property(lambda self: self._params.TSR)
    Uh = property(lambda self: self._params.Uh)
    Zh = property(lambda self: self._params.Zh)
    WV = property(lambda self: self._params.WV)
    Nv = property(lambda self: self._params.Nv)
    Ct = property(lambda self: self._params.Ct)
    Cp = property(lambda self: self._params.Cp)
    phi = property(lambda self: self._params.phi)
    dphi = property(lambda self: self._params.dphi)
    yloc = property(lambda self: self._params.yloc)
    zloc = property(lambda self: self._params.zloc)

    def _current_local(self):
        """Snapshot of the state WindFarm.solve() mutates directly on this instance."""
        return LocalConditions(Uhub=self.Uhub, V=self.V, W=self.W, Uin=self.Uin,
                               calculation_domain=self.calculation_domain)

    def init_Uin(self):
        return tp.init_Uin(self._params)

    @property
    def a(self):
        return self._params.a

    @property
    def R(self):
        return self._params.R

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
        return tp.calculate_efficiency(self._params, self.Uhub3)

    def simulate_vortex_field(self, seed=None, total_steps=1000):
        self.vortex_field, self.vortex_exhausted = tp.simulate_vortex_field(
            self._params, self._current_local(), seed=seed, total_steps=total_steps)

    def initialize_wake_field(self, companions=(), kw0=None, wage0=None, ti0=None):
        """`companions` are the other rotors at this same streamwise station, as (TurbineParams, dy) pairs..."""
        self.vortex_field, self.dl = tp.initialize_wake_field(
            self._params, self.vortex_field, self._current_local(), companions,
            kw0=kw0, wage0=wage0, ti0=ti0)

    def calculate_deficit_field(self, max_steps=1500, companions=(), carried=(),
                                n_slots=None, read_at=None):
        """`carried` are rotors shed at EARLIER stations whose near wake may still be developing, as..."""
        self.wake_field = tp.calculate_deficit_field(
            self._params, self._current_local(), self.vortex_field, self.dl,
            max_steps=max_steps, companions=companions, carried=carried, n_slots=n_slots,
            read_at=read_at
        )

class WindFarm:
    def __init__(self, config):
        self.turbine_configs = config.WindFarm
        self.field_params = config.Field
        self.turbines = []

        self._construct_wind_farm()

    def get_grid(self, tolerance=None):
        """Return grid dimensions and layout of the wind farm."""
        if not self.turbines:
            return {'rows': 0, 'cols': 0, 'x_positions': [], 'y_positions': [], 'layout': None}

        if tolerance is None:
            tolerance = self.turbines[0].D / 3.0

        x_coords = [t.pos[0] for t in self.turbines]
        x_positions = self._cluster_positions(x_coords, tolerance)

        y_coords = [t.pos[1] for t in self.turbines]
        y_positions = self._cluster_positions(y_coords, tolerance)

        rows = len(x_positions)
        cols = len(y_positions)

        layout = [[None for _ in range(cols)] for _ in range(rows)]

        for idx, turbine in enumerate(self.turbines):
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
        """Cluster positions that are within tolerance distance."""
        if not positions:
            return []

        sorted_pos = sorted(positions)
        clusters = []
        current_cluster = [sorted_pos[0]]

        for pos in sorted_pos[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [pos]

        clusters.append(np.mean(current_cluster))

        return clusters

    def _find_closest_cluster(self, position, cluster_centers):
        """Find index of closest cluster center to given position."""
        distances = [abs(position - center) for center in cluster_centers]
        return distances.index(min(distances))

    def _get_calculation_domain(self):
        """Calculate the downstream domain for each turbine as the distance to the furthest downstream..."""
        if not self.turbines:
            return
        x_positions = [t.pos[0] for t in self.turbines]
        max_x = max(x_positions)
        buffer = self.field_params.min_X * self.turbines[0].D
        for t in self.turbines:
            t.calculation_domain = (max_x + buffer - t.pos[0])

    def _construct_wind_farm(self):
        ys = [float(t.pos[1]) for t in self.turbine_configs.turbines()]
        y_bounds = (min(ys), max(ys))
        _configs = list(self.turbine_configs.turbines())
        self.turbines = [Turbine(t_config, self.field_params, y_bounds, len(_configs))
                         for t_config in _configs]
        self.turbines = sorted(self.turbines, key=lambda t: t.pos[0])
        self._get_calculation_domain()

    def calculate_efficiency(self, verbose=False):
        """Farm power as a fraction of the same farm with every rotor unwaked."""
        num = den = 0.0
        for t in self.turbines:
            P, P0 = tp.power_and_reference(t._params, t.Uhub3)
            if verbose:
                print(f"Turbine at pos={t.pos} m, yaw={t.yaw}°: "
                      f"Efficiency = {P / P0 * 100:.2f} %")
            num += P
            den += P0
        total_eff = num / den
        if verbose:
            print(f"Wind Farm Power Ratio: {total_eff * 100:.2f} %")
        return total_eff

    def solve(self, verbose=False, all_frames=True):
        """March one field and one vortex cloud through the farm; see wake_solver."""
        return solve_single_field(self, verbose=verbose, all_frames=all_frames)

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
