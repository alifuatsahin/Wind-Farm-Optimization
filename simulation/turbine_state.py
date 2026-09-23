import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field


# Exponent in Ct(beta) = Ct(0)*cos(beta)**CT_YAW_EXP.
CT_YAW_EXP = 2.0


@dataclass
class TurbineParams:
    """Everything about a turbine that is determined by config alone -- never mutated by WindFarm.solve()."""
    pos: np.ndarray
    D: float
    Zhub: float
    yaw: float
    TSR: float
    Ct: float
    Cp: float  # yaw-adjusted power coefficient
    config: object  # raw TurbineConfig, kept for .config.Cp (nominal power) access
    field_params: object
    Uh: float
    Zh: float
    WV: float
    Nv: int
    phi: np.ndarray
    dphi: float
    yloc: np.ndarray
    zloc: np.ndarray
    calculation_domain: float = 0.0  # set once by WindFarm after layout is known

    @property
    def beta(self):
        return np.deg2rad(self.yaw)

    @property
    def R(self):
        return self.D / 2

    @property
    def Rv(self):
        return 0.1 * self.D

    @property
    def Yoffset(self):
        return 5 * np.sin(self.beta)

    @property
    def Ct_yawed(self):
        """Thrust coefficient at the current yaw.

        Zong & Porte-Agel (2020) Sec 3, fitting Bastankhah & Porte-Agel (2016):
        Ct(beta) = Ct(0) * cos(beta)**1.6. The thrust counterpart of the
        Cp * cos(beta)**1.88 correction already applied in local_conditions."""
        return self.Ct * np.cos(self.beta) ** CT_YAW_EXP

    @property
    def a(self):
        """Axial induction factor, Zong & Porte-Agel (2020) Eq (3.3):
        a = (1 - sqrt(1 - Ct/cos(beta))) / 2, with Ct evaluated AT the yaw angle.

        Using the unyawed Ct here makes a *rise* with yaw (0.276 -> 0.329 at 25 deg)
        when it should fall (-> 0.252), over-deepening yawed wakes by ~26%: the
        measured near-wake minimum 0.347 matched 1-2a from the unyawed Ct exactly."""
        return (1 - np.sqrt(1 - self.Ct_yawed / np.cos(self.beta))) / 2

    @property
    def dl(self):
        """Grid spacing in y direction. Equivalent to the value historically read off
        vortex_field[0].yloc after vortex evolution -- see turbine_physics.py for the
        verification note; not yet wired into Turbine (kept additive for now)."""
        return float(self.yloc[1, 0] - self.yloc[0, 0])


jax.tree_util.register_dataclass(
    TurbineParams,
    data_fields=["pos", "D", "Zhub", "yaw", "TSR", "Ct", "Cp", "Uh", "Zh", "WV",
                 "phi", "dphi", "yloc", "zloc", "calculation_domain"],
    meta_fields=["config", "field_params", "Nv"],
)


@dataclass
class LocalConditions:
    """The state that WindFarm.solve() overwrites once per turbine, per solve() pass."""
    Uhub: float
    V: np.ndarray
    W: np.ndarray
    Uin: np.ndarray


jax.tree_util.register_dataclass(
    LocalConditions,
    data_fields=["Uhub", "V", "W", "Uin"],
    meta_fields=[],
)


@dataclass
class VortexSimConfig:
    """Registered-pytree replacement for the SimpleNamespace adapter
    turbine_physics.simulate_vortex_field builds to call vortex_model.simulate_vortex_evolution.
    Nv is meta: it is only read as a plain Python int (never appears inside a traced array
    expression in vortex_model.py), so marking it static avoids treating it as a value that
    could legitimately vary under trace, with no actual effect on correctness either way here."""
    D: float
    Uhub: float
    dgamma: np.ndarray
    calculation_domain: float
    phi: np.ndarray
    beta: float
    Yoffset: float
    Zhub: float
    Nv: int
    V: np.ndarray
    W: np.ndarray
    yloc: np.ndarray
    zloc: np.ndarray


jax.tree_util.register_dataclass(
    VortexSimConfig,
    data_fields=["D", "Uhub", "dgamma", "calculation_domain", "phi", "beta", "Yoffset",
                 "Zhub", "V", "W", "yloc", "zloc"],
    meta_fields=["Nv"],
)


@dataclass
class DeficitFieldConfig:
    """Registered-pytree replacement for the SimpleNamespace adapter
    turbine_physics.calculate_deficit_field builds to call utils.NuT_model and
    model_solver.advance_wake_field."""
    pos: np.ndarray
    D: float
    Uhub: float      # rotor-averaged LOCAL inflow (waked); what the turbine actually sees
    Uin: np.ndarray
    Zhub: float
    U0: float = 0.0  # rotor-averaged UNDISTURBED inflow (freestream), i.e. Du et al.'s U0
    rotor_mask: np.ndarray = None  # rotor disc, for the near-wake deficit ramp
    Ct_eff: float = 0.0            # Ct_yawed / cos(beta), i.e. the Ct entering Zong Eq 3.3


jax.tree_util.register_dataclass(
    DeficitFieldConfig,
    data_fields=["pos", "D", "Uhub", "Uin", "Zhub", "U0", "rotor_mask", "Ct_eff"],
    meta_fields=[],
)


def pack_upstream_turbines(upstream_turbines, n_max):
    """Packs a variable-length list of upstream Turbine objects into fixed-size (n_max, always
    the same across every turbine in a farm) arrays for utils.NuT_model, so its per-step call
    inside Loop 2's lax.scan body doesn't trace a structurally different program (and therefore
    recompile) per distinct upstream turbine count. Padding values are never observed (up_mask
    gates them out via jnp.where inside NuT_model) -- chosen only to avoid a division by zero."""
    n = len(upstream_turbines)
    up_a = jnp.zeros(n_max)
    up_D = jnp.ones(n_max)
    up_pos_x = jnp.zeros(n_max)
    up_Uhub = jnp.ones(n_max)
    up_mask = jnp.zeros(n_max, dtype=bool)
    if n > 0:
        up_a = up_a.at[:n].set(jnp.array([t.a for t in upstream_turbines]))
        up_D = up_D.at[:n].set(jnp.array([t.D for t in upstream_turbines]))
        up_pos_x = up_pos_x.at[:n].set(jnp.array([t.pos[0] for t in upstream_turbines]))
        up_Uhub = up_Uhub.at[:n].set(jnp.array([t.Uhub for t in upstream_turbines]))
        up_mask = up_mask.at[:n].set(True)
    return up_a, up_D, up_pos_x, up_Uhub, up_mask


def make_turbine_params(config, field_params) -> TurbineParams:
    """Replicates the static portion of the current Turbine.__init__."""
    Nv = field_params.Nv
    phi = np.linspace(-np.pi, np.pi, Nv)
    dphi = abs(phi[1] - phi[0])

    beta = np.deg2rad(config.yaw)
    Cp = config.Cp * np.cos(beta) ** 1.88

    Ly = field_params.max_Y * config.D
    Lz = field_params.max_Z * config.D
    n_grids = field_params.n_grids
    Ny = max(2, int(Ly / (config.D / n_grids)))
    Nz = max(2, int(Lz / (config.D / n_grids)))
    if config.Zhub - Lz / 2 < 0:
        zlims = (0, Lz)
    else:
        zlims = (config.Zhub - Lz / 2, config.Zhub + Lz / 2)
    yloc, zloc = np.meshgrid(
        np.linspace(-Ly / 2, Ly / 2, Ny), np.linspace(*zlims, Nz), indexing='ij'
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
    )
