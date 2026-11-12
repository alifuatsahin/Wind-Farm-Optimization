import numpy as np
import jax
from dataclasses import dataclass, field


CT_YAW_EXP = 2.0

CP_YAW_EXP = 3.0


@dataclass
class TurbineParams:
    """Everything about a turbine fixed by config alone."""
    pos: np.ndarray
    D: float
    Zhub: float
    yaw: float
    TSR: float
    Ct: float
    Cp: float
    config: object
    field_params: object
    Uh: float
    Zh: float
    WV: float
    Nv: int
    phi: np.ndarray
    dphi: float
    yloc: np.ndarray
    zloc: np.ndarray
    skew: float = 0.0
    n_rotors: int = 1

    @property
    def beta(self):
        return np.deg2rad(self.yaw)

    @property
    def beta_aero(self):
        """Misalignment against the LOCAL flow, which an upstream steered wake has already turned by..."""
        # beta is positive CLOCKWISE seen from above and skew = arctan(Vbar/Ubar) positive
        # ANTICLOCKWISE, so the misalignment is their SUM: a +y cross-flow on a +beta rotor
        # increases it, and an inflow along the rotor axis (skew = -beta) gives zero.
        return self.beta + self.skew

    @property
    def R(self):
        return self.D / 2

    @property
    def Yoffset(self):
        """Lateral shift of the rotor centre when yawed, Zong & Porte-Agel (2020) Eq (4.10): the rotor..."""
        return -float(self.config.overhang_D) * self.D * np.sin(self.beta)

    @property
    def Ct_yawed(self):
        """Thrust coefficient at the current yaw."""
        return self.Ct_at(self.beta_aero)

    def Ct_at(self, beta):
        """Ct at an ARBITRARY yaw."""
        return self.Ct * np.cos(beta) ** CT_YAW_EXP

    @property
    def a(self):
        cb = np.cos(self.beta)
        return (1 - np.sqrt(np.maximum(1 - self.Ct_yawed / cb, 0.0))) / 2

    def a_at(self, beta):
        """Axial induction factor, Zong & Porte-Agel (2020) Eq (3.3): a = (1 - sqrt(1..."""
        cb = np.cos(beta)
        return (1 - np.sqrt(np.maximum(1 - self.Ct_at(beta) / cb, 0.0))) / 2

    @property
    def dl(self):
        """Grid spacing in y direction."""
        return float(self.yloc[1, 0] - self.yloc[0, 0])


jax.tree_util.register_dataclass(
    TurbineParams,
    data_fields=["pos", "D", "Zhub", "yaw", "TSR", "Ct", "Cp", "Uh", "Zh", "WV",
                 "phi", "dphi", "yloc", "zloc", "skew"],
    meta_fields=["config", "field_params", "Nv", "n_rotors"],
)


@dataclass
class LocalConditions:
    """The state that WindFarm.solve() overwrites once per turbine, per solve() pass."""
    Uhub: float
    V: np.ndarray
    W: np.ndarray
    Uin: np.ndarray
    calculation_domain: float = 0.0


jax.tree_util.register_dataclass(
    LocalConditions,
    data_fields=["Uhub", "V", "W", "Uin", "calculation_domain"],
    meta_fields=[],
)


@dataclass
class VortexSimConfig:
    """Registered-pytree replacement for the SimpleNamespace adapter..."""
    D: float
    Uhub: float
    dgamma: np.ndarray
    calculation_domain: float
    phi: np.ndarray
    beta: float
    Yoffset: float
    Zhub: float
    nu_slope: float
    nu_const: float
    U_conv: float
    Nv: int
    V: np.ndarray
    W: np.ndarray
    yloc: np.ndarray
    zloc: np.ndarray
    pm_Av: np.ndarray = None
    pm_Aw: np.ndarray = None
    pm_lo: float = 0.0
    pm_lnr: float = 0.0


jax.tree_util.register_dataclass(
    VortexSimConfig,
    data_fields=["D", "Uhub", "dgamma", "calculation_domain", "phi", "beta", "Yoffset",
                 "Zhub", "nu_slope", "nu_const", "U_conv", "V", "W", "yloc", "zloc",
                 "pm_Av", "pm_Aw", "pm_lo", "pm_lnr"],
    meta_fields=["Nv"],
)


@dataclass
class DeficitFieldConfig:
    """Registered-pytree replacement for the SimpleNamespace adapter..."""
    pos: np.ndarray
    D: float
    Uhub: float
    Uin: np.ndarray
    Zhub: float
    U0: float = 0.0
    rotor_mask: np.ndarray = None
    Ct_eff: float = 0.0
    x_off: np.ndarray = None


jax.tree_util.register_dataclass(
    DeficitFieldConfig,
    data_fields=["pos", "D", "Uhub", "Uin", "Zhub", "U0", "rotor_mask", "Ct_eff",
                 "x_off"],
    meta_fields=[],
)


def make_turbine_params(config, field_params, y_bounds=None,
                        n_rotors=1) -> TurbineParams:
    """Replicates the static portion of the current Turbine.__init__."""
    Nv = field_params.Nv
    phi = np.linspace(-np.pi, np.pi, Nv, endpoint=False)
    dphi = abs(phi[1] - phi[0])

    beta = np.deg2rad(config.yaw)
    Cp = config.Cp * np.cos(beta) ** CP_YAW_EXP

    y_lo, y_hi = (config.pos[1], config.pos[1]) if y_bounds is None else y_bounds
    span_D = float(np.ceil((y_hi - y_lo) / config.D - 1e-9))
    Ly = span_D * config.D + field_params.max_Y * config.D
    Lz = field_params.max_Z * config.D
    n_grids = field_params.n_grids
    Ny = max(2, int(round(Ly / config.D * n_grids)))
    Nz = max(2, int(round(field_params.max_Z * n_grids)))
    if config.Zhub - Lz / 2 < 0:
        _dz = Lz / Nz
        zlims = (_dz / 2.0, Lz - _dz / 2.0)
    else:
        zlims = (config.Zhub - Lz / 2, config.Zhub + Lz / 2)
    y_centre = 0.5 * (y_lo + y_hi)
    yloc, zloc = np.meshgrid(
        np.linspace(y_centre - Ly / 2, y_centre + Ly / 2, Ny) - config.pos[1],
        np.linspace(*zlims, Nz), indexing='ij'
    )

    return TurbineParams(
        pos=config.pos,
        D=config.D,
        Zhub=config.Zhub,
        yaw=config.yaw,
        TSR=config.TSR,
        Ct=config.Ct,
        Cp=Cp,
        config=config,
        field_params=field_params,
        Uh=field_params.Uh,
        Zh=field_params.Zh,
        WV=field_params.WV,
        Nv=Nv,
        phi=phi,
        dphi=dphi,
        yloc=yloc,
        zloc=zloc,
        n_rotors=n_rotors,
    )
