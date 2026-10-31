import datetime
import os

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
    #: Rotor overhang (tower-to-rotor spacing) as a fraction of D. Per-turbine
    #: geometry: Jonkman et al. (2009) Table 1-1 gives 5.0 m on the NREL 5 MW's
    #: D = 126 m, i.e. 0.0397. The 0.2 default is the historical module constant and
    #: is NOT sourced -- replace it per rotor from the spec sheet.
    overhang_D: float = 0.2
    #: Nacelle/hub diameter as a fraction of D, for the co-located bluff-body disc.
    #: Per-turbine HARDWARE, and it must be ZERO for an actuator-disc simulation, which
    #: has no central body at all -- that is the discrimination the term passes. Zong &
    #: Porte-Agel (2020) Sec 2 gives the WIRE-01 nacelle as a Maxon DCX10L, 10 mm on a
    #: 150 mm rotor, i.e. 0.0667. Default 0.0 so nothing gets a body it does not have.
    nacelle_D: float = 0.0

@dataclass
class WindFarmConfig:
    pos: Optional[np.ndarray] = None  # Explicit turbine positions (x, y, z); leave unset to use 'grid' instead
    D: np.ndarray = field(default_factory=lambda: np.array([126.0]))  # Rotor diameter(s)
    Zhub: np.ndarray = field(default_factory=lambda: np.array([90.0]))  # Hub height(s)
    Ct: np.ndarray = field(default_factory=lambda: np.array([0.8]))  # Thrust coefficient(s)
    Cp: np.ndarray = field(default_factory=lambda: np.array([0.47]))  # Power coefficient(s)
    # Length 1 so it broadcasts to ANY farm size. It used to be np.array([0.0, 0.0, 0.0]),
    # which made a 1- or 2-turbine farm built from `pos` raise outright ("Cannot broadcast
    # array of size 3 to size 2") unless the caller passed yaw explicitly. For a 3-rotor
    # farm, the default grid, this broadcasts to the same [0, 0, 0] as before.
    yaw: np.ndarray = field(default_factory=lambda: np.array([0.0]))  # Yaw angle(s)
    TSR: np.ndarray = field(default_factory=lambda: np.array([7.02]))  # Tip speed ratio(s)
    overhang_D: np.ndarray = field(default_factory=lambda: np.array([0.2]))  # Rotor overhang / D
    nacelle_D: np.ndarray = field(default_factory=lambda: np.array([0.0]))  # Nacelle diameter / D

    elevation_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = field(default=None, repr=False)
    grid: InitVar[tuple] = (1,3,5,5)  # Optional grid definition: (n_rows, n_cols, spacing_x, spacing_y); ignored if 'pos' is given
    dist_type: str = "D"  # Distance type between the turbines: 'm (meters)' or 'D (x/D normalized over rotor diameters)'

    def __post_init__(self, grid):
        self.D = np.asarray(self.D, dtype=float)
        D_ref = self.D[0]

        # Explicit 'pos' always takes priority over 'grid' -- this also makes save_yaml/load_yaml
        # round-trip correctly, since a reloaded config always has concrete positions already resolved.
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

        # broadcast scalars or shorter arrays to match number of turbines (allows tiling)
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
            overhang_D = float(self._pick(self.overhang_D, idx)),
            nacelle_D = float(self._pick(self.nacelle_D, idx)),
        )
    
    def _broadcast(self, arr):
        """Expand a length-1 array to one value per turbine; pass length-n through.

        Every per-turbine field is validated against len(pos) by _check_consistency
        first, so only those two cases can reach here. There used to be a third branch
        that TILED a short array (`[a, b]` -> `[a, b, a]`) behind a printed warning,
        which silently gave the third rotor the first one's hardware; the check now
        rejects that instead, so the branch is gone rather than merely unreachable.
        """
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
        
        # overhang_D and nacelle_D are checked too: both are per-turbine HARDWARE, and
        # _broadcast TILES a short array behind nothing but a printed warning. For
        # nacelle_D that is a trap -- it must be exactly zero for an actuator-disc rotor
        # (see the nacelle carve in turbine_physics), and a length-2 array on a 3-rotor
        # farm would silently hand the third rotor a body it does not have.
        # Checked against the NUMBER OF TURBINES, not against each other. Comparing the
        # parameters only to max(lengths) let a lone odd-length array become the maximum
        # and pass: with every other field length 1, a length-2 nacelle_D on a 3-rotor
        # farm satisfied "1 or max_len" and was then TILED to [a, b, a] by _broadcast,
        # behind nothing but a printed warning.
        #
        # overhang_D and nacelle_D are included because both are per-turbine HARDWARE.
        # For nacelle_D the tiling is a trap: it must be exactly zero for an actuator-disc
        # rotor (see the nacelle carve in turbine_physics), and silently handing a rotor a
        # body it does not have is precisely what invalidates that term's justification.
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
    #: Ring elements shed per rotor. Bounded ABOVE by merging, not by cost: the yawed ring
    #: is foreshortened to semi-minor axis R*cos(beta), so its closest neighbour spacing is
    #: pi*D*cos(beta)/Nv and the ring survives only while that exceeds merge_threshold*D
    #: (see wake_solver._assert_ring_survives_merging). The old 49 satisfied that at 0 deg
    #: and FAILED from 40 deg, which is inside the +-40 deg yaw sweep -- 32 of its 117 rows
    #: were silently computed with a ring that collapsed on step one.
    #:
    #: 40 holds to 49 deg. Lowering it is free rather than a trade: measured at Nv =
    #: 36 / 44 / 49 the bench agrees to 0.0003 in every R2 (8-turbine, both LES, all three
    #: full-wake) and deflection RMSE moves at most 0.001 D, so the answer is converged
    #: well below the old value. merge_threshold is the wrong side of the inequality to
    #: move: it is a physics lever (raising it INCREASES deflection), Nv is discretisation.
    Nv: int = 40
    z0: float = 0.03 # Surface roughness length (Open sea 0.0002, Flat land 0.03)
    min_X: float = 7.0 # Minimum downstream distance to simulate (in rotor diameters)
    max_Y: float = 3.0 # Maximum lateral distance to simulate (in rotor diameters)
    max_Z: float = 2.0 # Maximum vertical distance to simulate (in rotor diameters)
    n_grids: int = 20 # Number of grids in each direction
    cfl_factor: float = 0.125 # CFL factor for adaptive time stepping
    merge_threshold: float = 0.05 # Vortex merging distance, IN ROTOR DIAMETERS.
    vortex_nu: str = 'abl'
    #: Rotor count at or above which the cloud's induction on the field grid is evaluated
    #: by the particle-mesh solver instead of the direct pairwise sum. None disables it.
    #:
    #: The direct sum is O(N*M) and is 86-92% of a cloud step at every farm size, so it is
    #: THE scaling term. The PM solve replaces it with a fixed number of FFTs, which is a
    #: large win at farm scale (80 rotors: 2313 ms -> ~90 ms) and a LOSS below a handful of
    #: rotors, because the bin count has to be static under jit: the ladder must span every
    #: core the whole run produces (measured 23x on the 3-rotor case, 34x on the 8-rotor
    #: row), so it costs ~76 transforms whether or not a given step needs them, against a
    #: direct sum that only costs 15 ms at that size.
    #:
    #: The threshold therefore exists to keep the VALIDATED cases on the exact kernel. PM
    #: is not bit-identical -- depositing vortices onto cells costs ~0.05% even with exact
    #: cores, and the ladder costs ~1% at pm_tol 0.05 -- so every published 3- and
    #: 8-rotor number stays on the direct path and is unchanged by construction.
    pm_min_rotors: int | None = None
    #: Relative width of the geometric core ladder the PM solver bins vortices onto.
    #: Measured field error against the direct sum: 4.70% at 0.20, 1.89% at 0.10, 0.95% at
    #: 0.05, 0.68% at 0.02. Cost is linear in the bin count, which goes as 1/ln(1+pm_tol).
    pm_tol: float = 0.05
    #: Friction velocity in m/s for the ABL vortex-core diffusivity. None derives it from
    #: the log law, u* = kappa*Uh/ln(Zh/z0), which is the historical behaviour and leaves
    #: every result unchanged.
    #:
    #: It is separable because z0 does DOUBLE DUTY and the two jobs want different numbers.
    #: z0 sets the inflow profile shape, and through turbine_physics.abl_nu_slope it also
    #: sets how fast the CVP decays and therefore how far a yawed wake steers. For the
    #: WIRE-01 tunnel those pull opposite ways: the back-scaled z0 = 0.03*(0.125/90) gives
    #: u* = 0.245 against Zong & Porte-Agel (2020) Sec 2's MEASURED 0.22 m/s, i.e. 11% high,
    #: while their measured z0 = 0.7 mm would give u* = 0.378, i.e. 72% high. So the
    #: back-scaled roughness is wrong for the profile and nearly right for the friction
    #: velocity, and substituting the measured roughness FLIPS THE SIGN of the steering gain
    #: (+1.63% -> -0.93%) by inflating u*, not by being more correct.
    #:
    #: Setting u_star directly is therefore the honest route: it is a measurable Zong
    #: reports, and it decouples the CVP decay from a roughness length that was never
    #: measured for this inflow. See memory z0-double-duty-flips-the-gain.
    u_star: Optional[float] = None

@dataclass
class Config:
    WindFarm: WindFarmConfig = field(default_factory=WindFarmConfig)
    Field: FieldConfig = field(default_factory=FieldConfig)

    # File output
    out_dir: str = "runs"             # root directory all runs are stored under
    run_name: Optional[str] = None   # explicit run folder name; auto-timestamped if left as None
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
            out_dir=data.get('out_dir', "runs"),
            run_name=data.get('run_name'),
        )

    def __repr__(self):
        return f"Config(Turbines={len(self.WindFarm)}, Uh={self.Field.Uh}, Out={self.out_path})"