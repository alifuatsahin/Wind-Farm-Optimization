import numpy as np
from dataclasses import dataclass, field
from typing import Iterator

@dataclass
class TurbineConfig:
    pos: np.ndarray     # shape (3,) or (x,y,z)
    D: float
    Zhub: float
    Ct: float
    yaw: float
    TSR: float

@dataclass
class WindFarmConfig:
    pos: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0, 0.0], [770.0, 0.0, 0.0], [1540.0, 0.0, 0.0]]))  # Turbine positions (x, y, z)
    D: np.ndarray = field(default_factory=lambda: np.array([77.0]))  # Rotor diameter(s)
    Zhub: np.ndarray = field(default_factory=lambda: np.array([80.0]))  # Hub height(s)
    Ct: np.ndarray = field(default_factory=lambda: np.array([0.75]))  # Thrust coefficient(s)
    yaw: np.ndarray = field(default_factory=lambda: np.array([0.0]))  # Yaw angle(s)
    TSR: np.ndarray = field(default_factory=lambda: np.array([7.0]))  # Tip speed ratio(s)

    def _broadcast(self, arr):
        a = np.asarray(arr)
        n = len(self)
        if len(a) == 1:
            return np.full(n, a[0])
        elif len(a) < n:
            return np.tile(a, (n // len(a) + 1))[:n]
        raise ValueError(f"Cannot broadcast array of size {a.size} to size {n}.")

    def check_consistency(self):
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError(f"Inconsistent shape for 'pos': {self.pos.shape}, expected (N, 3).")
        
        lengths = [len(self.D), len(self.Zhub), len(self.Ct), len(self.yaw), len(self.TSR)]
        max_len = max(lengths)
        for name, length in zip(['D', 'Zhub', 'Ct', 'yaw', 'TSR'], lengths):
            if length != 1 and length != max_len:
                raise ValueError(f"Inconsistent lengths in WindFarmConfig: '{name}' has length {length}, expected 1 or {max_len}.")
            
    def __post_init__(self):
        self.pos = np.atleast_2d(np.asarray(self.pos, dtype=float))
        # ensure numeric fields are numpy arrays
        self.D    = np.asarray(self.D, dtype=float)
        self.Zhub = np.asarray(self.Zhub, dtype=float)
        self.Ct   = np.asarray(self.Ct, dtype=float)
        self.yaw  = np.asarray(self.yaw, dtype=float)
        self.TSR  = np.asarray(self.TSR, dtype=float)

        self.check_consistency()

        # broadcast scalars or shorter arrays to match number of turbines (allows tiling)
        self.D    = self._broadcast(self.D)
        self.Zhub = self._broadcast(self.Zhub)
        self.Ct   = self._broadcast(self.Ct)
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
            yaw = float(self._pick(self.yaw, idx)),
            TSR = float(self._pick(self.TSR, idx)),
        )
    
    def turbines(self) -> Iterator[TurbineConfig]:
        for i in range(len(self)):
            yield self[i]

@dataclass(frozen=True)
class FieldConfig:
    Uh: float = 8.55 # Measured wind speed
    Zh: float = 80.0 # Height of the wind speed measurement
    WV: float = 0.0 # Vertical wind veer
    NuT_max: float = 0.03 # Maximum turbulence intensity
    Nv: int = 49
    z0: float = 0.5 # Surface roughness length
    max_X: float = 10.0 # Maximum downstream distance to simulate (in rotor diameters)
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