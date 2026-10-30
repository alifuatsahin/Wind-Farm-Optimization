import numpy as np
import jax
from dataclasses import dataclass, field


# Exponent in Ct(beta) = Ct(0)*cos(beta)**CT_YAW_EXP.
CT_YAW_EXP = 2.0

# Exponent in Cp(beta) = Cp(0)*cos(beta)**CP_YAW_EXP.
#
# 3 is the momentum-theory partner of CT_YAW_EXP = 2: the rotor-normal inflow is
# U*cos(beta), so T ~ (U cos b)^2 and P ~ (U cos b)^3.
#
# It is also the value Zong & Porte-Agel (2021) Renew. Energy 170, Eq (12) adopt for
# WIRE-01. Be precise about what that is, though: their Fig 9 regresses data from THREE
# OTHER studies' turbines (Adaramola & Krogstad 2011; Oabay et al. 2012; Bastankhah &
# Porte-Agel 2016), all "extracted at the optimal tip speed ratios", and the text states
# the resulting p = 1.8 and q = 3 "are under no means universal" and depend on the control
# scheme and blade geometry. So q = 3 is a sourced, principled choice that coincides with
# momentum theory -- NOT a measurement of this rotor. Fitting the full-wake dataset gives
# 2.42 at 25 deg rising to 2.9 at 40 deg, which the authors' own caveat anticipates.
#
# This was 1.88 for a long time, as a bare literal with no citation. 1.88 is Fleming et
# al. (2015), a NUMERICAL result for the NREL 5 MW -- a different turbine -- and it sits
# one hundredth from WIRE-01's THRUST exponent of 1.8, so it looks like the thrust
# coefficient was adopted as the power coefficient. Using 2 for thrust and 1.88 for power
# cannot both follow from momentum theory, and the three-turbine full-wake experiment
# rejects 1.88 outright: on the isolated upstream rotor, where no wake physics enters, it
# over-predicts retained power at every yaw angle and by 14 points at 40 deg.
#
# A least-squares fit to that experiment gives 2.82, not 3, and the fit is marginally
# better (sweep RMSE 0.0192 vs 0.0254). It is NOT used: a per-turbine parameter comes from
# the spec sheet even at a cost in R2. That rig also holds RPM roughly constant under yaw,
# so its rotor runs 5-14% above baseline tip-speed ratio and part of the measured loss is
# off-optimal operation rather than misalignment -- which biases the fitted exponent HIGH,
# since the extra loss makes the measured power fall faster than misalignment alone would.
# The true misalignment exponent is therefore even further below 3, not nearer to it.
# (This note previously said "biased low" and used that to support q = 3. The direction
# was wrong; the conclusion survives on the evidence below instead.)
#
# NO FIXED EXPONENT FITS. On T1 of the full-wake sweep, which is unwaked so its power is
# the rotor's own yaw response with no wake physics in the way, ln(P/P0)/ln(cos beta)
# reads 1.68 / 1.93 / 2.06 / 2.39 / 2.42 / 2.65 / 2.81 / 2.82 at 5 to 40 deg. The exponent
# climbs monotonically with angle, so any constant q is a compromise; 3 sits close across
# the useful range and is universal rather than per-turbine. Lowering it to 2.82 does NOT
# move the yaw optimum (verified exactly: Cp never enters the flow solution, so the whole
# sweep rescales analytically), and 2.70, which does move it, is further from the 2.85 the
# isolated rotor itself wants -- i.e. it would be buying the optimum by error cancellation.
#
# A DERIVED ALTERNATIVE WAS TESTED AND IS WORSE. Heck, Johlas & Howland (2023) JFM 959 A9
# (arXiv:2209.00111) extend momentum theory to a yawed disk and give P_r(beta) in closed
# form, Eqs (17)/(19) and the |v4|<<u4 forms (20)/(22), with no fitted constant; C_T' comes
# from the spec-sheet C_T by C_T' = C_T/(1-a)^2 = 1.617. Against the measured T1 curve it
# is 3.3-3.8x WORSE than cos^3 (RMSE 0.086 for the full Eq (15) system and 0.101 for the
# simplified form, against 0.026), over-predicting by 0.186 of baseline power at 40 deg.
# The cause is the premise the paper itself flags: it holds C_T' FIXED with yaw, whereas
# this rig holds RPM fixed, so TSR rises and both Cp and C_T' fall. Note this makes the
# comparison rig-specific: for a VARIABLE-SPEED rotor holding optimal TSR, cos^3 is exact
# by construction and Heck's induction correction puts the truth ABOVE it, so Heck may be
# the better model there. Neither dataset can settle that -- both tunnel cases are the same
# fixed-RPM rig and the LES measures no power. See memory: heck-yawed-disc-rejected.
#
# Strictly this is per-turbine and belongs in TurbineConfig, not here. It is a module
# constant only to mirror CT_YAW_EXP; move both together if a farm ever mixes rotors.
CP_YAW_EXP = 3.0


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
    #: Local inflow angle arctan(Vbar/Ubar) over this rotor's disc, radians. Set by
    #: wake_solver from the field the station is handed; 0 for the first station.
    skew: float = 0.0

    @property
    def beta(self):
        return np.deg2rad(self.yaw)

    @property
    def beta_aero(self):
        """Misalignment against the LOCAL flow, which an upstream steered wake has
        already turned by `skew`. The aerodynamics use this; the geometry (Yoffset,
        the projected disc, the vortex ring) keeps the commanded `beta`."""
        return self.beta - self.skew

    @property
    def R(self):
        return self.D / 2

    @property
    def Yoffset(self):
        """Lateral shift of the rotor centre when yawed, Zong & Porte-Agel (2020) Eq (4.10):
        the rotor sits l_n downstream of the tower, so yawing swings it sideways by
        l_n*sin(beta). Used as the centre of both the top-hat mask and the vortex ring,
        each of which is placed at -Yoffset.

        l_n is PER-TURBINE geometry and comes from TurbineConfig.overhang_D. It was a
        module constant 0.2 with no citation, which for the NREL 5 MW implies a 25.2 m
        overhang on a 126 m rotor; Jonkman et al. (2009) Table 1-1 gives 5.0 m, i.e.
        0.0397 D. At beta = 25 deg the two place the carve and the vortex ring 0.068 D
        apart -- nearly twice the case-A deflection RMSE -- and the spec value moves the
        yawed LES R2 0.89640 -> 0.90556. The aligned cases are bit-identical, as they
        must be: Yoffset ~ sin(beta). WIRE-01 keeps 0.2, which is equally uncited; 30 mm
        on a 150 mm rotor is plausible for that rig but nobody has sourced it.
        """
        return -float(self.config.overhang_D) * self.D * np.sin(self.beta)

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
        return self.Ct_at(self.beta_aero)

    def Ct_at(self, beta):
        """Ct at an ARBITRARY yaw. Everything that needs a yawed thrust coefficient goes
        through here, so CT_YAW_EXP cannot drift between call sites -- compute_Ut used to
        hardcode its own cos(beta)**1.6 and so ran on a different induction factor than
        the rest of the model."""
        return self.Ct * np.cos(beta) ** CT_YAW_EXP

    @property
    def a(self):
        # Numerator at the AERODYNAMIC angle, denominator the GEOMETRIC projected
        # area -- the combination that keeps 2a*piR^2cos(beta_geo) = T/(rho U).
        cb = np.cos(self.beta)
        return (1 - np.sqrt(np.maximum(1 - self.Ct_yawed / cb, 0.0))) / 2

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
                 "phi", "dphi", "yloc", "zloc", "skew"],
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
    #: Speed the cores age on. NOT Uhub -- see vortex_model's new_age.
    U_conv: float
    Nv: int
    V: np.ndarray
    W: np.ndarray
    yloc: np.ndarray
    zloc: np.ndarray


jax.tree_util.register_dataclass(
    VortexSimConfig,
    data_fields=["D", "Uhub", "dgamma", "calculation_domain", "phi", "beta", "Yoffset",
                 "Zhub", "nu_slope", "nu_const", "U_conv", "V", "W", "yloc", "zloc"],
    meta_fields=["Nv"],
)


@dataclass
class DeficitFieldConfig:
    """Registered-pytree replacement for the SimpleNamespace adapter
    turbine_physics.calculate_deficit_field builds to call utils.du_nu_t and
    model_solver.advance_wake_field."""
    pos: np.ndarray
    D: float
    Uhub: float      # rotor-averaged LOCAL inflow (waked); what the turbine actually sees
    Uin: np.ndarray
    Zhub: float
    U0: float = 0.0  # rotor-averaged UNDISTURBED inflow (freestream), i.e. Du et al.'s U0
    rotor_mask: np.ndarray = None  # rotor disc, for the near-wake deficit ramp
    Ct_eff: float = 0.0            # Ct_yawed / cos(beta), i.e. the Ct entering Zong Eq 3.3
    x_off: np.ndarray = None       # per-rotor streamwise age offset; see calculate_deficit_field


jax.tree_util.register_dataclass(
    DeficitFieldConfig,
    data_fields=["pos", "D", "Uhub", "Uin", "Zhub", "U0", "rotor_mask", "Ct_eff",
                 "x_off"],
    meta_fields=[],
)


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
    Cp = config.Cp * np.cos(beta) ** CP_YAW_EXP

    y_lo, y_hi = (config.pos[1], config.pos[1]) if y_bounds is None else y_bounds
    # span the farm laterally, plus half the margin at each end.
    #
    # The span is QUANTISED UP to a whole diameter, because Ny below is a jit-static
    # shape: an unquantised span gives a new Ny for every distinct lateral extent, so a
    # layout optimiser retraces on nearly every candidate. MEASURED on the space
    # run_optimization.position_optimization searches: 14 distinct Ny in 14 Sobol draws,
    # and on a new shape the compile is 2.03 s of a 4.09 s evaluation -- half the budget.
    # Quantising leaves one shape per whole diameter of span (5 over those draws).
    #
    # Rounding UP only ever widens the domain, never clips a wake, and it is inert for a
    # SINGLE ROW -- span 0 -> ceil(0) = 0 -- so every validation case in this repo is
    # bit-identical. The 1e-9 keeps an exact multiple from being bumped a diameter wider
    # by float noise.
    span_D = float(np.ceil((y_hi - y_lo) / config.D - 1e-9))
    Ly = span_D * config.D + field_params.max_Y * config.D
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
        # CELL-CENTRED at the ground: the first node sits at dz/2, not ON the wall.
        #
        # A node on a no-slip wall is where U vanishes by construction -- measured at
        # 0.0145 m/s against U0 = 4.87 -- and that costs twice. It is a SINK, because the
        # row is reset to Uin every step so whatever diffusion deposits there is thrown
        # away: the integrated momentum deficit lost 14.7% over 6-14 D, of which ~17
        # points is the boundary condition rather than physics. And it is a dt trap,
        # because anything that evolves the row divides by U ~ 0: max(nu_T/U) is 0.792
        # over the full plane against 0.0047 over the interior.
        #
        # dz/2 is inside the log layer, where Uin(z) is meaningful and the wall's effect
        # is carried by the profile as it should be. This is what finite-volume wall-
        # function codes do. Nz is unchanged, so the resolution is identical; only the
        # node positions shift by dz/2 and the top moves from Lz to Lz - dz/2.
        #
        # IT COULD NOT BE LANDED BEFORE. Measured earlier it took the 8-turbine row's T8
        # from .668 to .476 -- the sink was COMPENSATING the deep-array under-mixing, so
        # removing it exposed the real error. Re-measured after the incident-TI label was
        # allowed to spread with the wake (693e815), which raised nu_t over the outer
        # annulus: the deep array now IMPROVES, 8-turbine bias -.0032 -> -.0024 with
        # every rotor from T4 on moving toward unity, and nothing else moves by more than
        # 0.0003. The compensation is gone.
        _dz = Lz / Nz
        zlims = (_dz / 2.0, Lz - _dz / 2.0)
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
