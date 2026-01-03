import numpy as np
from dataclasses import asdict, dataclass, field, InitVar
from typing import Iterator, Callable, Optional
import yaml

from simulation.utils import npy_to_list

@dataclass
class TurbineConfig:
    pos: np.ndarray     # shape (3,) or (x,y,z)
    D: float
    Zhub: float
    Ct: float
    Cp: float
    yaw: float
    TSR: float

@dataclass
class WindFarmConfig:
    pos: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0, 0.0]]))  # Turbine positions (x, y, z)
    D: np.ndarray = field(default_factory=lambda: np.array([126.0]))  # Rotor diameter(s)
    Zhub: np.ndarray = field(default_factory=lambda: np.array([90.0]))  # Hub height(s)
    Ct: np.ndarray = field(default_factory=lambda: np.array([0.8]))  # Thrust coefficient(s)
    Cp: np.ndarray = field(default_factory=lambda: np.array([0.47]))  # Power coefficient(s)
    yaw: np.ndarray = field(default_factory=lambda: np.array([0.0]))  # Yaw angle(s)
    TSR: np.ndarray = field(default_factory=lambda: np.array([7.02]))  # Tip speed ratio(s)

    elevation_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = field(default=None, repr=False)
    grid: InitVar[tuple] = (3,3,5,5)  # Optional grid definition: (n_rows, n_cols, spacing_x, spacing_y)
    dist_type: str = "D"  # Distance type between the turbines: 'm (meters)' or 'D (x/D normalized over rotor diameters)'
            
    def __post_init__(self, grid):
        self.pos = np.atleast_2d(np.asarray(self.pos, dtype=float))
        self.D = np.asarray(self.D, dtype=float)

        # initialize positions from grid if specified
        D_ref = self.D[0]
        if grid:
            rows, cols, spacing_x, spacing_y = grid
            if len(self.D) > 1:
                print("WARNING: Initializing positions from grid with multiple rotor diameters. Using first turbine's D as reference.")
            scale = D_ref if self.dist_type == 'D' else 1.0
            x_coords = np.arange(cols) * spacing_x * scale
            y_coords = np.arange(rows) * spacing_y * scale
            xv, yv = np.meshgrid(x_coords, y_coords)
            self.pos = np.column_stack((xv.ravel(), yv.ravel(), np.zeros(xv.size)))
            self.dist_type = 'm'
        elif self.dist_type == 'D':
            self.pos *= D_ref
            self.dist_type = 'm'

        self.update_elevation()

        self.Zhub = np.asarray(self.Zhub, dtype=float)
        self.Ct   = np.asarray(self.Ct, dtype=float)
        self.Cp   = np.asarray(self.Cp, dtype=float)
        self.yaw  = np.asarray(self.yaw, dtype=float)
        self.TSR  = np.asarray(self.TSR, dtype=float)

        self._check_consistency()

        # broadcast scalars or shorter arrays to match number of turbines (allows tiling)
        self.D    = self._broadcast(self.D)
        self.Zhub = self._broadcast(self.Zhub)
        self.Ct   = self._broadcast(self.Ct)
        self.Cp   = self._broadcast(self.Cp)
        self.yaw  = self._broadcast(self.yaw)
        self.TSR  = self._broadcast(self.TSR)

    def __len__(self) -> int:
        return self.pos.shape[0]
    
    def _pick(self, arr, idx):
        a = np.asarray(arr)
        # scalar or single-element -> return scalar
        if a.size == 1:
            return a.flat[0]
        # array -> index
        return a[idx]
    
    def __getitem__(self, idx) -> TurbineConfig:
        # allow slice/int
        if isinstance(idx, slice):
            # return list of TurbineConfig for a slice
            return [self[i] for i in range(*idx.indices(len(self)))]
        n = len(self)
        if idx < 0:
            idx = n + idx
        if idx < 0 or idx >= n:
            raise IndexError("WindFarmConfig index out of range")
        return TurbineConfig(
            pos = np.asarray(self.pos[idx]),
            D = float(self._pick(self.D, idx)),
            Zhub = float(self._pick(self.Zhub, idx)),
            Ct = float(self._pick(self.Ct, idx)),
            Cp = float(self._pick(self.Cp, idx)),
            yaw = float(self._pick(self.yaw, idx)),
            TSR = float(self._pick(self.TSR, idx)),
        )
    
    def _broadcast(self, arr):
        a = np.asarray(arr)
        n = len(self)
        if len(a) == 1:
            return np.full(n, a[0])
        elif len(a) < n:
            print(f"WARNING: Broadcasting array of size {a.size} to size {n}.")
            return np.tile(a, (n // len(a) + 1))[:n]
        elif len(a) == n:
            return a
        raise ValueError(f"Cannot broadcast array of size {a.size} to size {n}.")

    def _check_consistency(self):
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError(f"Inconsistent shape for 'pos': {self.pos.shape}, expected (N, 3).")
        
        lengths = [len(self.D), len(self.Zhub), len(self.Ct), len(self.Cp), len(self.yaw), len(self.TSR)]
        max_len = max(lengths)
        for name, length in zip(['D', 'Zhub', 'Ct', 'Cp', 'yaw', 'TSR'], lengths):
            if length != 1 and length != max_len:
                raise ValueError(f"Inconsistent lengths in WindFarmConfig: '{name}' has length {length}, expected 1 or {max_len}.")
    
    def update_elevation(self):
        """Calculates and overrides the Z coordinate if elevation_func is provided."""
        if self.elevation_func is not None:
            x = self.pos[:, 0]
            y = self.pos[:, 1]
            self.pos[:, 2] = self.elevation_func(x, y)

    def turbines(self) -> Iterator[TurbineConfig]:
        for i in range(len(self)):
            yield self[i]

@dataclass(frozen=True)
class FieldConfig:
    Uh: float = 8.55 # Measured wind speed
    Zh: float = 80.0 # Height of the wind speed measurement
    WV: float = 0.0 # Vertical wind veer
    NuT_max: float = 0.05 # Maximum turbulent viscosity ratio
    I_amb: float = 0.072 # Ambient turbulence intensity
    Nv: int = 49
    z0: float = 0.03 # Surface roughness length (Open sea 0.0002, Flat land 0.03)
    max_X: float = 15.0 # Maximum downstream distance to simulate (in rotor diameters)
    max_Y: float = 3.0 # Maximum lateral distance to simulate (in rotor diameters)
    max_Z: float = 2.0 # Maximum vertical distance to simulate (in rotor diameters)
    n_grids: int = 20 # Number of grids in each direction
    cfl_factor: float = 0.125 # CFL factor for adaptive time stepping
    merge_threshold: float = 0.0075 # Vortex merging distance threshold

@dataclass
class Config:
    WindFarm: WindFarmConfig = field(default_factory=WindFarmConfig)
    Field: FieldConfig = field(default_factory=FieldConfig)

    # File output
    out_path: str = "Data/"

    def print(self):
        """Print only base dataclass fields (those with init=True)."""
        for name, f in self.__dataclass_fields__.items():
            if not f.init:
                continue
            val = getattr(self, name)
            # avoid printing large arrays fully
            if isinstance(val, np.ndarray):
                print(f"{name}: ndarray shape={val.shape}")
            else:
                print(f"{name}: {val}")

    def save_yaml(self, path: str):
        """Standard way to save configs."""
        # Helper to convert numpy to list for serialization
        with open(path, 'w') as f:
            yaml.dump(npy_to_list(asdict(self)), f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Map dictionary keys to class constructors
        return cls(
            WindFarm=WindFarmConfig(**data['WindFarm']),
            Field=FieldConfig(**data['Field']),
            out_path=data.get('out_path', "Data/")
        )

    def __repr__(self):
        return f"Config(Turbines={len(self.WindFarm)}, Uh={self.Field.Uh}, Out={self.out_path})"