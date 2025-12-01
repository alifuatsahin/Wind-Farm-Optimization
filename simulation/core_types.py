from .utils import smooth_2d, interpolate_vec_data, NuT_model
from .vortex_model import simulate_vortex_evolution
from .model_solver import advance_wake_field
from .superposition import get_local_velocity_field

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
        self.Ct = config.Ct
        self.yaw = config.yaw
        self.TSR = config.TSR
        self.Uh = field_params.Uh
        self.Vhub = 0.0
        self.Whub = 0.0
        self.Zh = field_params.Zh
        self.WV = field_params.WV
        self.Nv = field_params.Nv
        self.vortex_field = None  # to be filled after simulation
        self.wake_field = None  # to be filled after wake calculation
        self.dl = None  # grid spacing in y direction
        self._initialize_grid()

        self.Uhub = self._init_Uhub()
        self.Uin = self._init_Uin()

    def _initialize_grid(self):
        Ly = self.field_params.max_Y * self.D
        Lz = self.field_params.max_Z * self.D
        n_grids = self.field_params.n_grids

        Ny = max(2, int(Ly / (self.D / n_grids)))
        Nz = max(2, int(Lz / (self.D / n_grids)))
        self.yloc, self.zloc = np.meshgrid(np.linspace(-Ly/2, Ly/2, Ny), np.linspace(self.Zhub - Lz/2, self.Zhub + Lz/2, Nz), indexing='ij')

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
        gamma_ref = self.gamma0 * 0.44
        dgamma = gamma_ref / np.sum(dgamma[1:]) * dgamma  # Use np.sum
        return dgamma

    def _init_Uhub(self):
        Uhub = self.Uh * (np.log(self.Zhub / self.field_params.z0) / np.log(self.Zh / self.field_params.z0))
        return Uhub

    def _init_Uin(self):
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
    def phi(self):
        return np.linspace(-np.pi, np.pi, self.Nv)

    @property
    def dphi(self):
        return abs(self.phi[1] - self.phi[0])

    @property
    def Ut(self):
        return self._compute_Ut()

    @property
    def dgamma(self):
        return self._compute_dgamma()
    
    def calculate_power_output(self):
        P = 0.5 * 1.225 * 0.82 * np.mean(self.wake_field[0].U) ** 2 * (np.pi * (self.R ** 2))
        return P

    def simulate_vortex_field(self):
        self.vortex_field = simulate_vortex_evolution(self, self.field_params)
    
    def initialize_wake_field(self):
        yloc = self.vortex_field[0].yloc
        zloc = self.vortex_field[0].zloc
        beta = self.beta
        
        self.dl = float(yloc[1, 0] - yloc[0, 0])
        U = self.Uin.copy()
        # U = np.zeros_like(yloc)
        mask = np.sqrt(((yloc + self.Yoffset)**2) / (np.cos(beta)**2) + (zloc - self.Zhub)**2) <= self.R
        U[mask] -= 2.0 * self.Uhub * self.a

        hub_mask = (np.abs(yloc) <= self.dl * 1.0) & (zloc < self.Zhub)
        U[hub_mask] -= 0.3 * self.Uhub  # add some velocity deficit at the hub

        U_smooth = smooth_2d(U, kernel_size=3)
        self.vortex_field[0].U = U_smooth
        self.vortex_field[0].X = 0.0
        self.vortex_field[0].t = 0.0
        self.vortex_field[0].Uhub = self.Uhub

    def calculate_deficit_field(self, max_steps=10000):
        # Time-marching reduced-order model
        self.wake_field = [self.vortex_field[0]]
        dt = min(self.dl / self.Uhub, 0.25 * self.D / self.Uhub)
        max_X = self.field_params.max_X * self.D

        while self.wake_field[-1].X <= max_X:
            NuT = NuT_model(self.wake_field[-1].X, self, self.field_params)

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

    def _construct_wind_farm(self):
        self.turbines = [Turbine(t_config, self.field_params) for t_config in self.turbine_configs.turbines()]
        self.turbines = sorted(self.turbines, key=lambda t: t.pos[0])

    def calculate_power_output(self):
        total_power = 0.0
        for t in self.turbines:
            total_power += t.calculate_power_output()
        return total_power

    def simulate_turbine_vortex_fields(self):
        for t in self.turbines:
            print(f"Simulating turbine at pos={t.pos} m, yaw={t.yaw}°")
            t.simulate_vortex_field()

    def solve_streamwise(self):
        for t in self.turbines:
            U_local, V_local, W_local = get_local_velocity_field(t, self)

            # Create a mask for the rotor disk
            R = t.D / 2.0
            dist_from_hub = np.sqrt((t.yloc)**2 + (t.zloc - t.Zhub)**2)
            rotor_mask = dist_from_hub <= R
            
            t.Uhub = np.mean(U_local[rotor_mask])
            t.Vhub = np.mean(V_local[rotor_mask])
            t.Whub = np.mean(W_local[rotor_mask])
            t.Uin = U_local
            
            t.initialize_wake_field()
            t.calculate_deficit_field()

    def save_results(self, out_path):
        os.makedirs(out_path, exist_ok=True)
        for i, t in enumerate(self.turbines):
            rows = []
            for fi, frame in enumerate(t.wake_field):
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