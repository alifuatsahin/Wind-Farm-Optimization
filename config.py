import datetime
import os

import numpy as np
from dataclasses import asdict, dataclass, field, InitVar
from typing import Iterator, Callable, Optional
import yaml

from simulation.utils import npy_to_list

@dataclass
class TurbineConfig:
    pos: np.ndarray # shape (3,) or (x,y,z)
    D: float
    Zhub: float
    Ct: float
    Cp: float
    yaw: float
    TSR: float
    #: Rotor overhang (tower-to-rotor spacing) as a fraction of D.
    overhang_D: float = 0.2
    #: Nacelle/hub diameter as a fraction of D, for the co-located bluff-body disc.
    nacelle_D: float = 0.0

@dataclass
class WindFarmConfig:
    pos: Optional[np.ndarray] = None # Explicit turbine positions (x, y, z); leave unset to use 'grid' instead
    D: np.ndarray = field(default_factory=lambda: np.array([126.0])) # Rotor diameter(s)
    Zhub: np.ndarray = field(default_factory=lambda: np.array([90.0])) # Hub height(s)
    Ct: np.ndarray = field(default_factory=lambda: np.array([0.8])) # Thrust coefficient(s)
    Cp: np.ndarray = field(default_factory=lambda: np.array([0.47])) # Power coefficient(s)
    yaw: np.ndarray = field(default_factory=lambda: np.array([0.0])) # Yaw angle(s)
    TSR: np.ndarray = field(default_factory=lambda: np.array([7.02])) # Tip speed ratio(s)
    overhang_D: np.ndarray = field(default_factory=lambda: np.array([0.2])) # Rotor overhang / D
    nacelle_D: np.ndarray = field(default_factory=lambda: np.array([0.0])) # Nacelle diameter / D

    #: f(x, y) -> ground elevation, which overrides each turbine's z. None keeps z flat.
    elevation_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = field(default=None, repr=False)
    grid: InitVar[tuple] = (1,3,5,5) # Optional grid definition: (n_rows, n_cols, spacing_x, spacing_y); ignored if 'pos' is given
    dist_type: str = "D" # Distance type between the turbines: 'm (meters)' or 'D (x/D normalized over rotor diameters)'

    def __post_init__(self, grid):
        self.D = np.asarray(self.D, dtype=float)
        D_ref = self.D[0]

        if self.pos is not None:
            self.pos = np.atleast_2d(np.asarray(self.pos, dtype=float))
            if self.dist_type == 'D':
                self.pos = self.pos * D_ref
                self.dist_type = 'm'
        elif grid:
            rows, cols, spacing_x, spacing_y = grid
            if len(self.D) > 1:
                print("WARNING: Initializing positions from grid with multiple rotor diameters. Using first turbine's D as reference.")
            scale = D_ref if self.dist_type == 'D' else 1.0
            x_coords = np.arange(cols) * spacing_x * scale
            y_coords = np.arange(rows) * spacing_y * scale
            xv, yv = np.meshgrid(x_coords, y_coords)
            self.pos = np.column_stack((xv.ravel(), yv.ravel(), np.zeros(xv.size)))
            self.dist_type = 'm'
        else:
            raise ValueError("WindFarmConfig requires either 'pos' or 'grid' to be specified.")

        self.update_elevation()

        self.Zhub = np.asarray(self.Zhub, dtype=float)
        self.Ct   = np.asarray(self.Ct, dtype=float)
        self.Cp   = np.asarray(self.Cp, dtype=float)
        self.yaw  = np.asarray(self.yaw, dtype=float)
        self.TSR  = np.asarray(self.TSR, dtype=float)
        self.overhang_D = np.asarray(self.overhang_D, dtype=float)
        self.nacelle_D = np.asarray(self.nacelle_D, dtype=float)

        self._check_consistency()

        self.D    = self._broadcast(self.D)
        self.Zhub = self._broadcast(self.Zhub)
        self.Ct   = self._broadcast(self.Ct)
        self.Cp   = self._broadcast(self.Cp)
        self.yaw  = self._broadcast(self.yaw)
        self.TSR  = self._broadcast(self.TSR)
        self.overhang_D = self._broadcast(self.overhang_D)
        self.nacelle_D = self._broadcast(self.nacelle_D)

    def __len__(self) -> int:
        return self.pos.shape[0]

    def _pick(self, arr, idx):
        a = np.asarray(arr)
        if a.size == 1:
            return a.flat[0]
        return a[idx]

    def __getitem__(self, idx) -> TurbineConfig:
        if isinstance(idx, slice):
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
            overhang_D = float(self._pick(self.overhang_D, idx)),
            nacelle_D = float(self._pick(self.nacelle_D, idx)),
        )

    def _broadcast(self, arr):
        """Expand a length-1 array to one value per turbine; pass length-n through."""
        a = np.asarray(arr)
        n = len(self)
        if len(a) == 1:
            return np.full(n, a[0])
        if len(a) == n:
            return a
        raise ValueError(f"Cannot broadcast array of size {a.size} to size {n}.")

    def _check_consistency(self):
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError(f"Inconsistent shape for 'pos': {self.pos.shape}, expected (N, 3).")

        names = ['D', 'Zhub', 'Ct', 'Cp', 'yaw', 'TSR', 'overhang_D', 'nacelle_D']
        lengths = [len(getattr(self, n)) for n in names]
        max_len = len(self.pos)
        for name, length in zip(names, lengths):
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
    Uh: float = 10.0 # Measured wind speed
    Zh: float = 80.0 # Height of the wind speed measurement
    WV: float = 0.0 # Vertical wind veer
    I_amb: float = 0.072 # Ambient turbulence intensity
    #: Ring elements shed per rotor.
    Nv: int = 40
    z0: float = 0.03 # Surface roughness length (Open sea 0.0002, Flat land 0.03)
    min_X: float = 7.0 # Minimum downstream distance to simulate (in rotor diameters)
    max_Y: float = 3.0 # Maximum lateral distance to simulate (in rotor diameters)
    max_Z: float = 3.0 # Height of the vertical domain (in rotor diameters); converged from 3, not at 2
    n_grids: int = 20 # Number of grids in each direction
    cfl_factor: float = 0.125 # CFL factor for adaptive time stepping
    merge_threshold: float = 0.05 # Vortex merging distance, IN ROTOR DIAMETERS.
    #: Vortex-core diffusivity model: 'abl' (sheared, u*-based) or 'squire'.
    vortex_nu: str = 'abl'
    #: Rotor count at or above which the cloud's induction on the field grid is evaluated by the
    #: particle-mesh solver instead of the direct pairwise sum.
    pm_min_rotors: int | None = 16
    #: Relative width of the geometric core ladder the PM solver bins vortices onto.
    pm_tol: float = 0.05
    #: Friction velocity in m/s for the ABL vortex-core diffusivity.
    u_star: Optional[float] = None

@dataclass
class Config:
    WindFarm: WindFarmConfig = field(default_factory=WindFarmConfig)
    Field: FieldConfig = field(default_factory=FieldConfig)

    out_dir: str = "runs" # root directory all runs are stored under
    run_name: Optional[str] = None # explicit run folder name; auto-timestamped if left as None
    run_prefix: Optional[str] = None # prepended to the auto-generated timestamp; ignored if run_name is given

    def __post_init__(self):
        if self.run_name is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.run_name = f"{self.run_prefix}_{timestamp}" if self.run_prefix else timestamp

    @property
    def out_path(self) -> str:
        """Directory this run's figures/ and results/ subfolders are saved under."""
        return os.path.join(self.out_dir, self.run_name)

    def ensure_run_dir(self) -> str:
        """Create this run's output directory and drop a config.yaml snapshot in it, if not already done."""
        os.makedirs(self.out_path, exist_ok=True)
        config_path = os.path.join(self.out_path, "config.yaml")
        if not os.path.exists(config_path):
            self.save_yaml(config_path)
        return self.out_path

    def print(self):
        """Print only base dataclass fields (those with init=True)."""
        for name, f in self.__dataclass_fields__.items():
            if not f.init:
                continue
            val = getattr(self, name)
            if isinstance(val, np.ndarray):
                print(f"{name}: ndarray shape={val.shape}")
            else:
                print(f"{name}: {val}")

    def save_yaml(self, path: str):
        """Standard way to save configs."""
        with open(path, 'w') as f:
            yaml.dump(npy_to_list(asdict(self)), f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            WindFarm=WindFarmConfig(**data['WindFarm']),
            Field=FieldConfig(**data['Field']),
            out_dir=data.get('out_dir', "runs"),
            run_name=data.get('run_name'),
        )

    def __repr__(self):
        return f"Config(Turbines={len(self.WindFarm)}, Uh={self.Field.Uh}, Out={self.out_path})"
