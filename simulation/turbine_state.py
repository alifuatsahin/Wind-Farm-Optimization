import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field


# Exponent in Ct(beta) = Ct(0)*cos(beta)**CT_YAW_EXP.
CT_YAW_EXP = 2.0

# Tower-to-rotor spacing as a fraction of D.
L_NACELLE_D = 0.2


@dataclass
class TurbineParams:
    """Everything about a turbine fixed by config alone. Genuinely immutable once built:
    WindFarm.solve() never writes to it. calculation_domain used to live here and was
    written per segment by the march, which quietly broke that invariant -- it is mutable
    per-solve state and now sits in LocalConditions."""
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

    @property
    def beta(self):
        return np.deg2rad(self.yaw)

    @property
    def R(self):
        return self.D / 2

    @property
    def Yoffset(self):
        """Lateral shift of the rotor centre when yawed, Zong & Porte-Agel (2020) Eq (4.10):
        the rotor sits l_n downstream of the tower, so yawing swings it sideways by
        l_n*sin(beta). Used as the centre of both the top-hat mask and the vortex ring,
        each of which is placed at -Yoffset.
        """
        return -L_NACELLE_D * self.D * np.sin(self.beta)

    @property
    def Ct_yawed(self):
        """Thrust coefficient at the current yaw.

        DEVIATION FROM ZONG, deliberate. His Eq (3.3) uses Ct(beta) = Ct(0)*cos(beta)**1.6,
        which he states is a FIT to Bastankhah & Porte-Agel (2016) data, not a derivation.
        CT_YAW_EXP = 2 is instead the momentum-theory value: the rotor-normal inflow is
        U*cos(beta), so T = 2*rho*A*a*(1-a)*(U cos beta)^2 and Ct(beta) = Ct(0)cos^2(beta).
        Backing the exponent out of the yawed LES gives 1.97. It also makes
        Ct_eff = Ct*cos(beta), the combination that appears throughout the yawed-wake
        literature. Note Heck, Johlas & Howland (2023) JFM 959 argue no fixed cos^n is
        exact, since the induction itself depends on both yaw and Ct."""
        return self.Ct_at(self.beta)

    def Ct_at(self, beta):
        """Ct at an ARBITRARY yaw. Everything that needs a yawed thrust coefficient goes
        through here, so CT_YAW_EXP cannot drift between call sites -- compute_Ut used to
        hardcode its own cos(beta)**1.6 and so ran on a different induction factor than
        the rest of the model."""
        return self.Ct * np.cos(beta) ** CT_YAW_EXP

    @property
    def a(self):
        return self.a_at(self.beta)

    def a_at(self, beta):
        """Axial induction factor, Zong & Porte-Agel (2020) Eq (3.3):
        a = (1 - sqrt(1 - Ct(beta)/cos(beta))) / 2, with Ct evaluated AT the yaw angle.
        Under CT_YAW_EXP = 2 this collapses to a = (1 - sqrt(1 - Ct(0)cos(beta)))/2.

        Ct(beta) must be the yawed value: using the unyawed Ct makes a RISE with yaw
        (0.276 -> 0.329 at 25 deg) when it should fall, over-deepening yawed wakes."""
        cb = np.cos(beta)
        return (1 - np.sqrt(np.maximum(1 - self.Ct_at(beta) / cb, 0.0))) / 2

    @property
    def dl(self):
        """Grid spacing in y direction. Equivalent to the value historically read off
        vortex_field[0].yloc after vortex evolution -- see turbine_physics.py for the
        verification note; not yet wired into Turbine (kept additive for now)."""
        return float(self.yloc[1, 0] - self.yloc[0, 0])


jax.tree_util.register_dataclass(
    TurbineParams,
    data_fields=["pos", "D", "Zhub", "yaw", "TSR", "Ct", "Cp", "Uh", "Zh", "WV",
                 "phi", "dphi", "yloc", "zloc"],
    meta_fields=["config", "field_params", "Nv"],
)


@dataclass
class LocalConditions:
    """The state that WindFarm.solve() overwrites once per turbine, per solve() pass."""
    Uhub: float
    V: np.ndarray
    W: np.ndarray
    Uin: np.ndarray
    calculation_domain: float = 0.0  # how far downstream this rotor's field is marched


jax.tree_util.register_dataclass(
    LocalConditions,
    data_fields=["Uhub", "V", "W", "Uin", "calculation_domain"],
    meta_fields=[],
)


@dataclass
class VortexSimConfig:
    """Registered-pytree replacement for the SimpleNamespace adapter
    turbine_physics.simulate_vortex_field builds to call vortex_model._simulate_vortex_evolution_jit.
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
    nu_slope: float
    nu_const: float
    Nv: int
    V: np.ndarray
    W: np.ndarray
    yloc: np.ndarray
    zloc: np.ndarray


jax.tree_util.register_dataclass(
    VortexSimConfig,
    data_fields=["D", "Uhub", "dgamma", "calculation_domain", "phi", "beta", "Yoffset",
                 "Zhub", "nu_slope", "nu_const", "V", "W", "yloc", "zloc"],
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


def make_turbine_params(config, field_params, y_bounds=None) -> TurbineParams:
    """Replicates the static portion of the current Turbine.__init__.

    `y_bounds` is the (min, max) lateral position of EVERY rotor in the farm. The
    cross-plane grid is built to span that whole extent plus the usual max_Y margin,
    and is therefore shared in absolute y by every turbine. The single marched field
    reads turbine i's inflow by interpolating turbine i-1's field onto i's grid, so
    if the two grids do not overlap the interpolation falls back to the undisturbed
    profile and the wake is silently lost -- a farm with two columns 7D apart used to
    report every rotor as unwaked. Sharing the grid makes the chain correct for any
    layout. With one turbine, or a single row, y_bounds collapses and the grid is
    exactly the +-max_Y*D/2 it has always been.
    """
    Nv = field_params.Nv
    phi = np.linspace(-np.pi, np.pi, Nv, endpoint=False)
    dphi = abs(phi[1] - phi[0])

    beta = np.deg2rad(config.yaw)
    Cp = config.Cp * np.cos(beta) ** 1.88

    y_lo, y_hi = (config.pos[1], config.pos[1]) if y_bounds is None else y_bounds
    # span the farm laterally, plus half the margin at each end
    Ly = (y_hi - y_lo) + field_params.max_Y * config.D
    Lz = field_params.max_Z * config.D
    n_grids = field_params.n_grids
    # D cancels algebraically here (Ly = max_Y*D), so take the ratio directly rather
    # than round-tripping through it: (max_Z*D)/(D/n_grids) evaluates to
    # 17.999999999999996 for D = 12.6, and the truncation then silently drops a grid
    # point and shifts results by ~0.5%. Whether it happens depends on the floating-
    # point representation of D alone, which is not something a user can anticipate.
    # resolution is held fixed at n_grids points per diameter, so the point count
    # grows with the span rather than the span being resampled onto a fixed count
    Ny = max(2, int(round(Ly / config.D * n_grids)))
    Nz = max(2, int(round(field_params.max_Z * n_grids)))
    if config.Zhub - Lz / 2 < 0:
        zlims = (0, Lz)
    else:
        zlims = (config.Zhub - Lz / 2, config.Zhub + Lz / 2)
    # yloc stays LOCAL to this rotor (the rotor disc sits at yloc = 0), but the
    # absolute positions yloc + pos[1] are identical for every turbine in the farm.
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
    )
