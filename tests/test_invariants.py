"""Dimensional and physical invariants of the wake model.

Runs standalone (`python tests/test_invariants.py`) or under pytest if installed.

These are the cheapest tests that would have caught real bugs. Two of the six found
while developing this model were absolute lengths left at wind-tunnel scale -- a
vortex-merging threshold and the tower-to-rotor offset, both fixed to D = 0.15 m --
which are invisible at the scale they were tuned on and silently wrong at any other.
`test_geometric_similarity` catches exactly that class of error.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, FieldConfig, WindFarmConfig
from simulation import Simulation
from simulation.turbine_state import CP_YAW_EXP

YAW_CASES = ([0.0, 0.0], [25.0, 0.0])


def _farm(D, zh, Uh, yaw, n=2, spacing_D=6.0, n_grids=12):
    """A small farm specified entirely in diameters, so geometry scales with D."""
    wf = WindFarmConfig(grid=(1, n, spacing_D, spacing_D), D=np.array([D]),
                        Zhub=np.array([zh]), Ct=np.array([0.8]), Cp=np.array([0.47]),
                        yaw=np.asarray(yaw, dtype=float)[:n], TSR=np.array([7.0]),
                        dist_type='D')
    # z0 scales with zh so that log(zh/z0) -- the only shape in the inflow profile --
    # is held fixed as the geometry is scaled.
    fc = FieldConfig(Uh=Uh, Zh=zh, z0=zh * 3.3e-4, I_amb=0.08, min_X=6.0, max_Y=2.0,
                     max_Z=1.5, n_grids=n_grids)
    sim = Simulation(Config(WindFarm=wf, Field=fc, run_prefix='_test'))
    sim.run()
    return sim


def _normalised_inflows(sim):
    return np.array([t.Uhub for t in sim.wind_farm.turbines]) / sim.wind_farm.field_params.Uh


def test_geometric_similarity():
    """Scaling every length by 10 at fixed speed must leave normalised results unchanged.

    The model has no molecular viscosity and no intrinsic length scale, so D, z_hub,
    spacing and z0 can be scaled together with no effect on U/Uh. Any hard-coded length
    -- a merge distance in metres, a nacelle offset in metres -- breaks this.
    """
    for yaw in YAW_CASES:
        small = _normalised_inflows(_farm(D=12.6, zh=9.0, Uh=8.0, yaw=yaw))
        large = _normalised_inflows(_farm(D=126.0, zh=90.0, Uh=8.0, yaw=yaw))
        assert np.allclose(small, large, rtol=2e-6, atol=2e-6), (
            f'not scale invariant at yaw={yaw}: {small} vs {large}')


def test_velocity_scaling():
    """Scaling the reference speed must leave U/Uh unchanged: the model is linear in Uh."""
    for yaw in YAW_CASES:
        slow = _normalised_inflows(_farm(D=126.0, zh=90.0, Uh=5.0, yaw=yaw))
        fast = _normalised_inflows(_farm(D=126.0, zh=90.0, Uh=15.0, yaw=yaw))
        assert np.allclose(slow, fast, rtol=1e-9, atol=1e-9), (
            f'not velocity scale invariant at yaw={yaw}: {slow} vs {fast}')


def test_shed_circulation_sums_to_zero():
    """The hub vortex must cancel the tip ring: no net streamwise circulation.

    Required by the absence of body forces outside the rotor. A sign or normalisation
    error in the shed-circulation split shows up here at once. Only the real vortices
    are summed -- the array's second half is the ground-image system, which negates the
    first and would make the total vanish regardless.
    """
    sim = _farm(D=126.0, zh=90.0, Uh=8.0, yaw=[25.0], n=1)
    circ = np.asarray(sim.wind_farm.turbines[0].vortex_field.Circ)
    circ = circ[0] if circ.ndim > 1 else circ
    real = circ[:len(circ) // 2]          # exclude the mirror images
    scale = np.abs(real).sum()
    assert scale > 0, 'no circulation was shed at all'
    assert abs(real.sum()) / scale < 1e-9, (
        f'net circulation {real.sum():.3e} is not zero (scale {scale:.3e})')


def test_yaw_trades_own_power_for_the_rotor_behind():
    """Yawing the upstream rotor must cost it power and relieve the one behind it.

    This is the trade-off the optimiser exists to exploit. If the objective did not
    contain it, the optimum would sit at the yaw bound for trivial reasons.
    """
    eff0 = [t.calculate_efficiency()
            for t in _farm(126.0, 90.0, 8.0, [0.0, 0.0]).wind_farm.turbines]
    eff25 = [t.calculate_efficiency()
             for t in _farm(126.0, 90.0, 8.0, [25.0, 0.0]).wind_farm.turbines]
    assert eff25[0] < eff0[0], f'yaw did not cost the front rotor power: {eff0} -> {eff25}'
    assert eff25[1] > eff0[1], f'yaw did not help the rotor behind: {eff0} -> {eff25}'


def _row(pos, max_Y=3.0, ct=0.8):
    """A farm at explicit absolute positions, in metres.

    `ct` may be a scalar or one value per rotor, so a test can make a rotor inert."""
    n = len(pos)
    wf = WindFarmConfig(pos=np.array(pos, float), D=np.array([80.0]),
                        Zhub=np.array([70.0]), Ct=np.atleast_1d(np.asarray(ct, float)),
                        Cp=np.array([0.47]),
                        yaw=np.zeros(n), TSR=np.array([7.0]), dist_type='m')
    fc = FieldConfig(Uh=8.0, Zh=70.0, z0=0.0002, I_amb=0.077, max_Y=max_Y, min_X=7.0,
                     max_Z=2.0, n_grids=14, Nv=20)
    sim = Simulation(Config(WindFarm=wf, Field=fc, run_prefix='_test'))
    sim.run()
    return {(round(t.pos[0] / 80.0), round(t.pos[1] / 80.0)): t.Uhub / 8.0
            for t in sim.wind_farm.turbines}


def test_lateral_neighbour_does_not_disturb_a_column():
    """A rotor far to the side must not change what a column sees.

    Two failures used to hide here. The cross-plane grid was sized per rotor, so a
    second column simply fell outside it and the interpolation returned the undisturbed
    profile -- a 2x2 farm reported every rotor unwaked. And the march was cut at every
    rotor in x, so a neighbour at the same station shortened the segment to 0.5D and the
    upstream wake stopped developing, worth +17.9%.
    """
    alone = _row([[0, 0, 0], [560, 0, 0]])[(7, 0)]
    with_neighbour = _row([[0, 0, 0], [0, 560, 0], [560, 0, 0]])[(7, 0)]
    err = abs(with_neighbour - alone) / alone
    assert err < 0.01, (f'a rotor 7D to the side moved the column by {100*err:.1f}% '
                        f'({alone:.6f} -> {with_neighbour:.6f})')


def test_distant_upstream_rotor_does_not_change_a_column():
    """A rotor to the side AND upstream must not change a column it never touches.

    The sibling test above puts its neighbour at the SAME x as the column's first rotor, so
    any quantity referenced to "the farm's first rotor" is unchanged and the test is blind to
    a whole class of defect. Here A is 7D to the side and 5D further upstream, so it becomes
    the farm's first rotor without ever entering the column.

    Measured: the pure lateral leak (A moved onto B's station) is 0.22% for every closure, the
    numerical floor, so nothing transmits sideways. But a farm-referenced budget length scale
    moved the probe by 12.26% -- worse than the 9.325% that disqualified Crespo -- because
    x_farm is a streamwise distance irrespective of lateral position. The station-local
    reference gives 0.17% and the width floor 1.74%.

    Crespo PASSES this at 0.12%: its non-locality is purely the streamwise station grouping,
    and a rotor 7D to the side is gated out by its own |dy| < 3D criterion, so it never
    contributes. The bound is a real budget -- a farm-referenced budget age failed it at
    12.26%, while the grouping invariant rated that same option at 0.014%.
    """
    base = _row([[400, 0, 0], [800, 0, 0]])[(10, 0)]
    with_a = _row([[0, 560, 0], [400, 0, 0], [800, 0, 0]])[(10, 0)]
    err = abs(with_a - base) / base
    assert err < 0.01, (
        f'a rotor 7D to the side and 5D upstream moved the column by {100*err:.2f}% '
        f'({base:.6f} -> {with_a:.6f}): the budget length scale is reading rotor positions '
        f'rather than the local flow')


def test_inert_rotor_on_the_axis_does_not_change_the_rotor_behind():
    """A thrustless rotor ON the wake axis must change nothing. It is not a flow feature.

    The two probes above both put the bystander to the SIDE, so neither can see a defect
    that needs the bystander to sit inside the wake. This one does: a rotor at Ct = 0.001
    extracts no momentum and sheds no circulation, so the column downstream of it must be
    the column it would be without it. Anything else means a STATION'S EXISTENCE is
    changing the answer, which disqualifies the model for layout optimisation -- the
    optimiser is free to put a rotor anywhere, including directly in a wake it then
    discovers is a bad place for one.

    It failed at 2.6% (Ct-independent, so entirely the station and not the thrust) for as
    long as Du Eq (26)'s age and the incident TI were read fresh at every station: the
    bystander created a station, the station restarted the age for the WHOLE plane, and
    the eddy viscosity downstream started from zero age instead of five diameters of it.
    That is the same class as the 9.325% grouping discontinuity that disqualified Crespo,
    and 25x the bound here. The fix is to carry both as blended fields -- see
    initialize_wake_field and AGE_BLEND_EPS.

    The bound is a real budget: the lateral control (same bystander moved 7D aside, where
    the restart cannot act) measures 0.10%, so 0.1% is the floor this probe can resolve.
    """
    base = _row([[0, 0, 0], [800, 0, 0]], ct=[0.8, 0.8])[(10, 0)]
    with_inert = _row([[0, 0, 0], [400, 0, 0], [800, 0, 0]],
                      ct=[0.8, 0.001, 0.8])[(10, 0)]
    err = abs(with_inert - base) / base
    assert err < 0.005, (
        f'an inert rotor on the axis moved the rotor behind it by {100*err:.2f}% '
        f'({base:.6f} -> {with_inert:.6f}): a station is restarting the eddy-viscosity '
        f'age or the incident TI instead of carrying it')


def test_identical_rotors_see_identical_inflow():
    """In a 2x2 farm the two downstream rotors are geometrically identical."""
    d = _row([[0, 0, 0], [0, 560, 0], [560, 0, 0], [560, 560, 0]])
    a, b = d[(7, 0)], d[(7, 7)]
    assert abs(a - b) / a < 0.01, f'identical rotors disagree: {a:.6f} vs {b:.6f}'
    assert a < 0.9, f'downstream rotor is not waked at all ({a:.6f}): wake chain broken'

def test_isolated_yawed_turbine_loses_exactly_the_cosine_power():
    """An isolated turbine's efficiency must be exactly Cp(beta)/Cp(0) = cos^CP_YAW_EXP.

    The exponent is read from the model rather than written here, so the two cannot drift
    apart -- which they did. This asserted cos^1.88 against a hardcoded 1.88 for a long
    time, so it stayed green while the exponent itself was wrong: 1.88 is a NUMERICAL
    result for the NREL 5 MW, whereas WIRE-01's measured power exponent is 3. The test only
    ever checked that the two sides of the ratio used the same disc, never that the law was
    right. The three-turbine full-wake experiment is what caught it, on the isolated
    upstream rotor where no wake physics enters.

    It stands in undisturbed flow, so the disc-averaged inflow IS
    nominal_hub_velocity and the ratio collapses. Any departure means the two sides are
    being read over different discs. They were: the power disc sat at the tower axis
    while the initial condition, the vortex ring and nominal_hub_velocity all sat at the
    rotor centre, l_n*sin(beta) to the side -- a 0.085D mismatch at 25 deg that fed
    straight into the objective the optimiser maximises.
    """
    for beta in (0.0, 10.0, 20.0, 25.0, 30.0):
        wf = WindFarmConfig(pos=np.array([[0.0, 0.0, 0.0]]), D=np.array([80.0]),
                            Zhub=np.array([70.0]), Ct=np.array([0.8]),
                            Cp=np.array([0.47]), yaw=np.array([beta]),
                            TSR=np.array([7.0]), dist_type='m')
        fc = FieldConfig(Uh=8.0, Zh=70.0, z0=0.0002, I_amb=0.077, max_Y=3.0, min_X=7.0,
                         max_Z=2.0, n_grids=14, Nv=20)
        sim = Simulation(Config(WindFarm=wf, Field=fc, run_prefix='_test'))
        sim.run()
        eta = sim.calculate_objective()
        ref = np.cos(np.deg2rad(beta)) ** CP_YAW_EXP
        assert abs(eta - ref) / ref < 1e-12, (
            f'beta={beta}: eta={eta:.10f} but Cp ratio is {ref:.10f}; the power disc and '
            f'the reference disc disagree')

def test_wide_farm_marches_without_diverging():
    """A farm with two columns must march to the end with admissible velocities.

    The cross-plane had no lateral boundary condition: the only Dirichlet applied was
    U[:, 0] = Uin[:, 0], which on a (Ny, Nz) field is the GROUND, not the far field, so
    the y-edges were free. On a grid spanning a whole farm that free edge sits in
    nominally undisturbed flow and grows: a two-column farm reached U = -13 m/s in a
    corner cell. The global max(Us) in the step heuristic then latched onto it, the mask
    collapsed from 1438 cells to 9, and the march froze outright.
    """
    for cols, label in ((1, 'one column'), (2, 'two columns')):
        pos = [[r * 560.0, c * 560.0, 0.0] for c in range(cols) for r in range(8)]
        n = len(pos)
        wf = WindFarmConfig(pos=np.array(pos, float), D=np.array([80.0]),
                            Zhub=np.array([70.0]), Ct=np.array([0.8]),
                            Cp=np.array([0.47]), yaw=np.zeros(n),
                            TSR=np.array([7.0]), dist_type='m')
        fc = FieldConfig(Uh=8.0, Zh=70.0, z0=0.0002, I_amb=0.077, max_Y=3.0, min_X=7.0,
                         max_Z=2.0, n_grids=12, Nv=20)
        sim = Simulation(Config(WindFarm=wf, Field=fc, run_prefix='_test'))
        sim.run()        # a stall raises RuntimeError out of _march_segment
        worst = min(float(np.nanmin(np.asarray(t.wake_field[-1].U)))
                    for t in sim.wind_farm.turbines)
        assert worst > -1e-6, (
            f'{label}: velocity went inadmissible (min U = {worst:.3f} m/s), so the '
            f'march is diverging rather than merely deep')


# --------------------------------------------------------------------------------------
# Near-wake ramp: it must not matter WHERE the march happens to be cut.
#
# The ramp is applied as a ratio of consecutive ages, so over a march it telescopes to
# (1 - 2a(age_end)) / (1 - 2a(age_start)), and the initial condition is carved with exactly
# a(0). A full segment therefore composes to the correctly developed deficit. But the ramp
# stack holds only the CURRENT station's rotors and ages them all by the station-local
# march distance, so when the march advances, a rotor leaves the stack and its ramp freezes
# wherever it had reached. Cut a segment to L << 2D and its deficit is stuck near the
# half-strength carve value 1 - 2a(0).
#
# Measured cost: a 3x3 farm rotated 0.01 deg -- geometrically indistinguishable from
# aligned -- read 21% high when the rows split into separate stations (0.7709 vs 0.6375).
# Disabling the ramp entirely collapsed that gap to 2.8%, which is what identifies the ramp
# as the cause rather than the eddy viscosity (worth ~2%) or the grouping path (~2%).
# --------------------------------------------------------------------------------------

def _uhub_near(sim, x_m, y_m, tol_m=40.0):
    """Uhub/Uh of the rotor nearest (x_m, y_m). Positions here differ by less than a
    diameter, so the rounded-diameter keys `_row` builds would collide."""
    best = min(sim.wind_farm.turbines,
               key=lambda t: (t.pos[0] - x_m) ** 2 + (t.pos[1] - y_m) ** 2)
    assert abs(best.pos[0] - x_m) < tol_m and abs(best.pos[1] - y_m) < tol_m, (
        f'no rotor near ({x_m}, {y_m})')
    return best.Uhub / 8.0


def _sim(pos, yaw=None, n_grids=14):
    pos = np.array(pos, float)
    n = len(pos)
    yaw = np.zeros(n) if yaw is None else np.asarray(yaw, float)
    wf = WindFarmConfig(pos=pos, D=np.array([80.0]), Zhub=np.array([70.0]),
                        Ct=np.array([0.8]), Cp=np.array([0.47]), yaw=yaw,
                        TSR=np.array([7.0]), dist_type='m')
    fc = FieldConfig(Uh=8.0, Zh=70.0, z0=0.0002, I_amb=0.077, max_Y=3.0, min_X=7.0,
                     max_Z=2.0, n_grids=n_grids, Nv=20)
    sim = Simulation(Config(WindFarm=wf, Field=fc, run_prefix='_test'))
    sim.run()
    return sim


def test_ramp_composes_over_a_subdivided_march():
    """Ramping in N steps must equal ramping in one, for a rotor that keeps its own age.

    This is the identity the fix relies on: the per-step factor is a ratio of consecutive
    ages, so the product over any subdivision of [0, L] collapses to the endpoint ratio.
    It holds for the FUNCTION unconditionally -- what today's solver breaks is the
    premise, by resetting the age to zero at every station instead of carrying it. If this
    ever fails, the ratio form has been replaced by something non-composable and the whole
    per-rotor-offset approach is invalid.
    """
    from simulation.turbine_physics import axial_induction_ramped

    ct, L = 0.8, 2.5
    one = ((1.0 - 2.0 * axial_induction_ramped(ct, L))
           / (1.0 - 2.0 * axial_induction_ramped(ct, 0.0)))
    for n in (2, 7, 50):
        edges = np.linspace(0.0, L, n + 1)
        many = 1.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            many *= ((1.0 - 2.0 * axial_induction_ramped(ct, hi))
                     / (1.0 - 2.0 * axial_induction_ramped(ct, lo)))
        assert abs(float(many) - float(one)) < 1e-6, (
            f'ramp does not compose over {n} sub-steps: {float(many):.9f} vs '
            f'{float(one):.9f}')


def test_completed_and_padded_rotors_are_inert_in_the_ramp_stack():
    """Keeping old rotors in the stack must not perturb the cases already validated.

    The fix keeps every shed rotor in the ramp stack so it can go on developing across a
    station boundary. That is only safe if a rotor whose near wake is complete contributes
    a factor of exactly 1, and if the constant-length padding (Ct_eff = 0) does too --
    otherwise every validated row moves.

    Spacing matters and the margin is thinner than it looks. In the float32 the solver
    actually runs, erf saturates by 4D so the factor is exactly 1.0; under
    jax_enable_x64 it is 1 - 6e-13 at 5D and 1 - 5e-9 at 4D. The bound below is loose
    enough to hold in both, and tight enough that compounding over the 1500-step march
    budget stays near 1e-6 -- four orders below the tolerances the behavioural tests use.
    At 3D the factor is 0.999994 and a rotor is NOT inert, so this guarantee does not
    extend to spacings below about 4D.
    """
    from simulation.turbine_physics import axial_induction_ramped

    for age_D in (5.0, 7.0, 14.0):
        a_old = axial_induction_ramped(0.8, age_D)
        a_new = axial_induction_ramped(0.8, age_D + 0.05)
        f = float((1.0 - 2.0 * a_new) / (1.0 - 2.0 * a_old))
        assert abs(f - 1.0) < 1e-9, (
            f'a rotor {age_D}D upstream is not inert in the ramp stack: factor {f:.17g}')

    a_old = axial_induction_ramped(0.0, 0.0)
    a_new = axial_induction_ramped(0.0, 0.05)
    f = float((1.0 - 2.0 * a_new) / (1.0 - 2.0 * a_old))
    assert f == 1.0, f'a padded (Ct_eff = 0) slot is not inert: factor {f:.17g}'


def test_grouping_tolerance_does_not_change_the_answer():
    """X_STATION_TOL_D is a numerical parameter and must not move the physics.

    The geometry is held EXACTLY fixed and only the tolerance changes, so there is no
    modelling-error defence: any difference is the march reacting to where it was cut.
    Two rotors sit 7D apart laterally with a 0.1D streamwise offset, and a third sits 7D
    behind the first. Above the tolerance the two front rotors share a station and each
    gets a full 7D segment; below it they split and the first gets a 0.1D segment, so its
    ramp freezes at erf(0.1) and its wake stays near half strength.
    """
    import simulation.wake_solver as ws

    pos = [[0.0, 0.0, 0.0], [8.0, 560.0, 0.0], [560.0, 0.0, 0.0]]
    original = ws.X_STATION_TOL_D
    out = {}
    try:
        for tol in (0.25, 1e-6):
            ws.X_STATION_TOL_D = tol
            out[tol] = _uhub_near(_sim(pos), 560.0, 0.0)
    finally:
        ws.X_STATION_TOL_D = original

    grouped, split = out[0.25], out[1e-6]
    err = abs(split - grouped) / grouped
    assert err < 0.01, (
        f'the grouping tolerance changed the answer by {100*err:.1f}% on identical '
        f'geometry: grouped {grouped:.6f} vs split {split:.6f}')

    # ANCHOR. Agreement alone is not enough -- the two branches could meet on the wrong
    # value. The grouped branch is the trustworthy one here, because grouping is what
    # gives every rotor a full 7D segment and so a completed ramp, and the fix cannot
    # reach it (both front rotors end up ~7D upstream of the only other station, where a
    # shed rotor is inert). Pin it, so the test cannot be satisfied by both branches
    # drifting to the split branch's 0.91, which has T3 nearly unwaked.
    # This PASSES at 0.357% with the shipped closure, whose incident TI is read per cell
    # from the transported k_w field rather than looked up from rotor positions. The
    # now-removed Crespo & Hernandez closure FAILED here at 9.3%, and that discrepancy was
    # OWNED BY THE CLOSURE, not by the march: Crespo measures x from each UPSTREAM rotor, so
    # its nu_T depended on which rotors the station grouping put upstream. A closure that
    # reads nu_T from local fields cannot. Three points isolated it: Crespo with the old
    # non-conservative operator 9.3%, Crespo with the conservative operator 9.3%, Du Eq (26)
    # with the conservative operator 0.267%. The diffusion operator was NOT the cause.
    # This is the invariant that made the k_w-fed closure the default -- see also
    # test_distant_upstream_rotor_does_not_change_a_column, which catches an off-axis
    # non-locality this one is blind to.
    # Re-recorded 2026-10-18 with the compact diffusion stencil and the 2-D dt limit
    # (see golden_row). +0.173%, and the grouped-vs-split agreement is 0.353%, unchanged
    # from the 0.357% this closure was recorded at -- the locality property this anchor
    # exists to protect is untouched by the operator change.
    # Re-recorded 2026-10-23 when the ramp increment went from a uniform pedestal over the
    # rigid disc to the same increment weighted by the azimuthal mean of the wake deficit.
    # +1.368%. The locality this anchor exists to protect is untouched: the closure still
    # reads only local fields, and the weight is renormalised PER ROTOR over that rotor's
    # own mask, so it cannot couple a rotor to the station grouping either.
    # Re-recorded 2026-10-27 with the azimuthally weighted ramp removed in favour of
    # Zong's uniform multiplier; was 0.76910753494860562. The LOCALITY this anchor exists
    # to protect is unchanged -- grouping -0.348%, distant-upstream +0.048%.
    # Re-recorded the same day for the carried age/TI fields (63458af) and then for the
    # FRACTIONAL ROTOR CARVE; was 0.75872868698909457, i.e. -0.117% in total. Locality is
    # not merely unchanged but much better than when this anchor was last set: grouping
    # -0.348% -> +0.047%, threshold -0.908% -> -0.129%. The carve cannot couple a rotor to
    # the station grouping at all -- it is cell geometry, evaluated per rotor.
    # Re-recorded for COVERAGE-WEIGHTED rotor averages; was 0.75784234656775307, +0.123%.
    # The locality this anchor protects is untouched and the two LEAK probes improve
    # sharply: lateral +0.061% -> -0.009%, distant-upstream +0.050% -> -0.014%. A weight
    # is read per rotor from its own geometry, so it cannot couple a rotor to the station
    # grouping at all.
    # Re-recorded for the coverage-weighted RAMP FOOTPRINT; was 0.75877803471657956,
    # -0.261%. Locality is untouched -- grouping +0.044%, lateral -0.006%,
    # distant-upstream -0.014% -- and a coverage weight is read per rotor from its own
    # geometry, so it cannot couple a rotor to the station grouping.
    # Re-recorded for the (1-a)*Uhub core ageing; was 0.75679968436946754, i.e. -2.8e-8
    # -- the same aligned-row insensitivity as the golden row above.
    # Re-recorded for the cell-centred ground grid; was 0.75679966295046597, +0.082%.
    assert abs(grouped - 0.75741989645145347) / 0.75741989645145347 < 1e-9, (
        f'the grouped branch itself moved: {grouped:.17g}; re-record only if this was '
        f'deliberate, since it is the reference the split branch is held to')


def test_answer_is_continuous_across_the_grouping_threshold():
    """Farm output must be continuous in the layout, or layout optimisation is unusable.

    Same probe, but the tolerance is left alone and the streamwise offset is moved across
    it: 0.24D groups, 0.26D splits. The layouts differ by 0.02D = 1.6 m, which is
    physically nothing, so an optimiser walking a rotor through this offset must not see a
    step. This is the user-facing form of the test above -- it fails for the same reason
    and is kept because it is the statement that actually blocks a direction sweep.
    """
    near = _uhub_near(_sim([[0.0, 0.0, 0.0], [0.24 * 80.0, 560.0, 0.0],
                            [560.0, 0.0, 0.0]]), 560.0, 0.0)
    far = _uhub_near(_sim([[0.0, 0.0, 0.0], [0.26 * 80.0, 560.0, 0.0],
                           [560.0, 0.0, 0.0]]), 560.0, 0.0)
    err = abs(far - near) / near
    assert err < 0.01, (
        f'moving a rotor 0.02D across the grouping threshold changed a column by '
        f'{100*err:.1f}%: {near:.6f} -> {far:.6f}')


def test_validated_cases_are_unchanged_by_the_ramp_fix():
    """Golden values recorded before the fix, for cases it must leave alone.

    Every spacing here is >= 5D, where a carried rotor's ramp term is mathematically zero,
    so the fix cannot move these PHYSICALLY. It does move them in the last bits: carrying a
    rotor puts a second live mask in the tensordot and reassociates it, measured at 1.7e-13
    relative on T3-T5 (one float64 ULP if the 5D rotor is skipped outright instead). So the
    1e-9 below is a real budget, not slack -- it sits four orders above the reassociation
    noise and four below anything physical.

    NOTE these guard the RAMP change specifically. Any change to the eddy-viscosity closure
    or to the diffusion operator does shift the wake fringes and is EXPECTED to move the row
    values; when one lands, these goldens must be re-recorded deliberately, not relaxed. That
    has happened once, for the conservative operator -- see the note on golden_row.
    """
    # Re-recorded 2026-10-27 for COVERAGE-WEIGHTED rotor averages. These two SHOULD BE
    # EQUAL: the yawed disc is the unyawed one compressed in y by cos(beta), so every
    # z-row's y-extent scales by the same factor, the relative weighting across z is
    # untouched, and the disc mean of a z-only profile cannot change. Fine quadrature
    # confirms it to 9e-16 at 0.99649857303760725. Measured against that exact value the
    # boolean disc was +0.0443% at 0 deg and +0.1116% at 25; weighting gives +0.0075% and
    # -0.0102%, i.e. 6x and 11x closer, and the SPURIOUS YAW DEPENDENCE of an isolated
    # rotor's own inflow -- which fed every yaw comparison in the model -- falls from
    # 0.0671% to 0.0177%. See tp.disc_weights_on.
    # Re-recorded for the CELL-CENTRED ground grid. These two should be EQUAL and the
    # exact value is 0.99649857303760725 (see above); the new grid is 55x closer at 0 deg,
    # +0.00754% -> +0.00014%, and cuts the spurious yaw dependence a further 2.7x,
    # 0.01767% -> 0.00657%. That is the midpoint rule beating endpoint-inclusive
    # quadrature in z: moving the nodes off the boundaries removes the edge effect.
    golden_isolated = {0.0: 0.99649994412742249, 25.0: 0.99643426281798975}
    for beta, ref in golden_isolated.items():
        got = _sim([[0.0, 0.0, 0.0]], [beta]).wind_farm.turbines[0].Uhub / 8.0
        assert abs(got - ref) / ref < 1e-9, (
            f'isolated turbine at yaw={beta} moved: {got:.17g} vs golden {ref:.17g}')

    # These are the GAMMA_TRAILING_K = 0.45 values. They were briefly re-recorded for
    # k = 1.0 (T2 moved 0.006%, T5 0.086%, T1 stayed bit-identical) and then restored when
    # k went back to Zong's measured 0.45 -- see GAMMA_TRAILING_K's own note. That episode
    # is why this test records numbers rather than tolerating drift: it is the only thing
    # that would have caught the constant changing underneath the row cases.
    # The now-removed Crespo closure gave
    # [.., 0.68561188238556825, 0.70839363258543719, 0.69427784837223705, 0.68659866597497632].
    # Shipped-closure values (incident TI from the marched k_w). T1 and T2 match the Crespo
    # row to 4e-11 and are unchanged in substance: the incident TI at the FIRST rotor is
    # exactly ambient, which is both closures' bay-1 condition and Du's calibration condition.
    # The row then declines where Crespo's rises, which is the k_w budget saturating down a row
    # -- see [[crespo-validity-envelope]].
    # Re-recorded 2026-10-17 when the budget's nu_t and Psi moved from the farm-ambient TI
    # to the TI INCIDENT on each rotor, which is what Du's Eq (26)/(28) coefficients are
    # functions of. T1-T3 are UNCHANGED to the last digit and only T4-T5 move (+0.88%,
    # +1.23%), which is the consistency check that the change did what it claims: the
    # deficit already used ti_in, so bay 2's VELOCITY is untouched; only its k_w changes,
    # and that first reaches the velocity field one bay later, through ti_in at T3. For the
    # same reason the grouped 3-rotor anchor above did not move.
    #
    # Re-recorded 2026-10-18 for the DIFFUSION OPERATOR: utils.div_nu_grad / laplacian_
    # compact replaced two applications of jnp.gradient, and turbine_physics.DIFFUSIVE_CFL
    # replaced the one-dimensional dt limit with the two-dimensional one. Exactly the case
    # the docstring above says to re-record for, and the moves are the right size and
    # shape: T1 bit-identical (an isolated rotor's Uhub cannot see downstream diffusion),
    # T2-T5 at +0.25/-0.08/-0.04/+0.19%, i.e. a few tenths of a percent and NOT monotone --
    # a discretisation change redistributing the wake fringe, not a physics change, which
    # would push one way down the row. Cross-check: on the LES cases the compact stencil
    # converges (CFL 0.45) to R2 0.92635 against the old scheme's 0.9262, so the previous
    # answer was correct and this only changes how it is reached.
    #
    # Amended later the same day when the diffusive step limit was corrected a second
    # time, to bound max(nu_T/U) rather than max(nu_T) -- the pseudo-time the operator
    # sees is dx/U, not dt. T1-T2 are bit-identical and T3-T5 move +0.016/+0.036/+0.020%,
    # i.e. only the bays whose cores are deep enough for the U in that ratio to bind.
    #
    # Amended again when utils.advance_kw's diffusion term went to the conservative flux
    # form, div(nu_t grad k_w), the momentum equation's form having been carried through
    # first. T1-T3 are BIT-IDENTICAL and only T4-T5 move, +0.53/+0.47%, which is the
    # one-bay lag spelled out above arriving a second time: in bay 1 the budget's nu_t is
    # the scalar ambient TI, so grad(nu_t) = 0 and the two forms are the same equation
    # there; bay 2 is the first with a ti_in FIELD, so its k_w is the first to move, and
    # that k_w reaches a velocity through ti_in at T3, hence a Uhub at T4. Both moves are
    # UPWARD, i.e. toward the measurement, and the grouped anchor above did not move at all
    # -- it is a 3-rotor case, and T1-T3 are exactly the rotors this cannot reach.
    # Re-recorded 2026-10-23 for a PHYSICS change, the first of these re-records that is:
    # the ramp increment is no longer a uniform pedestal over the rigid disc but the same
    # increment weighted by the azimuthal mean of the wake's own deficit, i.e. a radially
    # varying a(r) inferred from the wake rather than from blade data. T2-T5 move
    # +1.18/+3.22/+1.84/+1.32%, an order of magnitude larger than any discretisation
    # re-record above, which is what a deliberate physics change should look like.
    #
    # T1 is BIT-IDENTICAL, and not only because an isolated rotor's Uhub cannot see its own
    # wake: where the deficit is radially uniform inside the disc the weight reduces to the
    # mask EXACTLY, since w = m*d*(sum m / sum m*d) = m when d is constant on the mask. The
    # change is a strict generalisation that does nothing to a uniformly loaded wake and
    # only acts once shear and diffusion have made the profile non-uniform -- which is also
    # why it improves the three real-rotor datasets and costs the LES actuator disc 0.0025.
    #
    # All five moves are UPWARD, i.e. downstream rotors see faster flow. That is the
    # narrower wake covering less of the disc Uhub averages over, and it is toward the
    # measurement: the 8-turbine row's P/Pref goes 0.82 -> 0.89 at T2 and its R2
    # 0.9235 -> 0.9304.
    # Amended 2026-10-26 for the aerodynamic-yaw change: a waked rotor is now
    # misaligned against the LOCAL flow, beta_aero = beta_geo - arctan(Vbar/Ubar),
    # rather than against the tunnel axis. T1 and T2 are BIT-IDENTICAL -- T1 has no
    # upstream wake and an aligned rotor's wake is laterally symmetric, so the
    # disc-averaged V at T2 is exactly zero. T3-T5 move 5.9e-6 / 2.4e-5 / 2.6e-5,
    # growing down the row: that is a small DISCRETISATION asymmetry in the lateral
    # grid, which the correction now converts into a negligible aerodynamic one via
    # cos^2(delta). It is 1e-5, far under any physical signal, but it does mean an
    # aligned farm is no longer exactly laterally symmetric in its aerodynamics --
    # worth watching if the lateral grid ever changes.
    # Re-recorded 2026-10-26 for utils.advance_kw: the budget now advects k_w at the
    # LOCAL U, as the momentum march does, instead of the scalar U0, and gets the
    # boundary condition it never had (zero flux through the ground, zero gradient on
    # the sides). T1 AND T2 are bit-identical, which is the one-bay lag in its purest
    # form -- bay 1's budget runs on the frozen ambient ti_in, so no change to k_w there
    # can reach the bay-1 velocity field at all, and T3 is the first rotor whose inflow
    # a k_w change can touch. T3-T5 move +0.97/+0.44/+0.35%, all UPWARD, i.e. toward the
    # measurement, as the 8-turbine model/measured power ratios do at every rotor
    # (T3 .891 -> .975, T5 .734 -> .741, T8 .601 -> .606).
    #
    # Amended the same day when k_w gained its cross-plane advection by (V, W), the term
    # the momentum equation has always carried. T1 and T2 stay bit-identical for the
    # reason above, and T3-T5 move only 2.7e-6 / 5.6e-6 / 6.1e-6 -- which is the point:
    # an ALIGNED row has no CVP, so its only cross-flow is the wake's own swirl, which is
    # antisymmetric over the disc and very nearly cancels in a rotor average. The term is
    # yaw-specific by construction, and this case is the control that shows it. Where it
    # does act, full-wake case A's deflection RMSE improves 0.0356 -> 0.0331 D.
    # Re-recorded 2026-10-26 for the UNLINEARISED dissipation: the budget now uses Du's
    # own physical form C_eps k_w^(3/2)/l_m instead of the linearised nu_t k_w / Psi he
    # substitutes only to make his Eq (13) analytically solvable by Green's function. We
    # march, so we do not need the linearity. T1 and T2 bit-identical (the one-bay lag
    # again), T3 -0.03%, then T4 +1.14% and T5 +1.65% -- GROWING down the row, which is
    # the signature of TI that now accumulates instead of declining. That is the point:
    # the equilibrium exponent changes from k ~ S^2 to k ~ S^(4/3), so the 6x deficit-
    # gradient collapse down a row costs 11x of TKE instead of 36x.
    # Re-recorded 2026-10-27, TWICE, both deliberate:
    #   * the vortex t_limit 1.5 -> 2.5, so the cloud covers the deficit march it steers.
    #     T1 bit-identical, T2-T5 moved 4e-8 to 1.3e-7 -- a longer vortex march perturbing
    #     the dt sequence, with every bench metric unchanged to 4 decimals.
    #   * the azimuthally weighted ramp REMOVED for Zong's uniform multiplier. This one is
    #     physical and large: T2 0.69543 -> 0.68731, T5 0.67449 -> 0.66613, i.e. a deeper
    #     row, since the weighting had been deepening the core at the edge's expense.
    # T1 is untouched by both -- an isolated rotor's own Uhub does not read its own wake.
    # Re-recorded 2026-10-27 for the SHEAR-LAYER AGE AND INCIDENT TI becoming blended
    # fields instead of per-station restarts. T1 AND T2 are bit-identical, which is the
    # one-bay lag in its cleanest form yet and is the consistency check on the change: the
    # first rotor has no old deficit to blend against, and bay 1's ti_in is frozen ambient,
    # so neither blend can act before the second station or reach a Uhub before the third
    # rotor. T3-T5 then move +5.02/+5.71/+5.96%, UPWARD and GROWING down the row -- age and
    # TI that accumulate down a row instead of restarting at every station. That is a
    # physics change and it is the largest re-record in this list, as it should be: the
    # defect it fixes was an inert Ct=0.001 rotor on the axis moving the rotor behind it by
    # 2.6%, i.e. a station's existence changing the answer. See AGE_BLEND_EPS for the
    # cut-off that regularises the blend and the window it was measured in.
    # Amended the same day when the incident TI stopped being frozen in POSITION as well
    # as in value -- utils.advect_incident_ti carries it with (V, W). T1 and T2 stay
    # bit-identical for the reason above, and T3-T5 move only -0.09/-0.21/-0.29%: an
    # ALIGNED row has no CVP, so its only cross-flow is the wake's own swirl, which is
    # antisymmetric over the disc and nearly cancels in a rotor average. This case is the
    # CONTROL for that change, exactly as it was for k_w's cross-plane advection; where
    # the term is meant to act, the yawed cases move 10-30x as far (LES yawed R2
    # .8804 -> .8823, full-wake B .9117 -> .9145, its TI R2 .751 -> .758).
    # Amended again for the FRACTIONAL ROTOR CARVE: the disc now removes momentum in
    # proportion to each cell's coverage instead of in/out by cell centre. T1 and the two
    # isolated cases above are BIT-IDENTICAL -- a rotor's own Uhub reads its inflow, not
    # its own wake -- and T2-T5 move -0.065/+0.053/+0.049/+0.051%. Small, and NOT an
    # accuracy change: the whole bench moves by <= 0.0007. It is landed for the SHAPE of
    # the error, not its size. See _disc_fraction.
    # Amended with golden_isolated above for the coverage-weighted averages: T1 -0.037%
    # (it is the isolated case), then +0.149/+0.059/+0.103/+0.110%.
    # Amended for the COVERAGE-WEIGHTED RAMP FOOTPRINT, the last of the three boolean
    # discs (carve d24cc24, rotor average 692468d, now the ramp). T1 and both isolated
    # cases are BIT-IDENTICAL -- the ramp acts only on a wake, and an isolated rotor's
    # Uhub reads its undisturbed inflow -- and T2-T5 fall a uniform 0.12-0.15%.
    # Amended for DIFFUSING the incident-TI label with nu_t, which lets it reach the wake
    # edge instead of stopping at the disc edge. T1 AND T2 are bit-identical, and so is
    # the grouped anchor above: bay 1's ti is uniform ambient and diffusing a constant
    # changes nothing, so the first effect cannot reach a Uhub before T3. Then T3 -0.32%,
    # T4 +0.21%, T5 +0.60% -- a sign flip and then growth, which is the outer annulus
    # entraining more as the wake widens past the disc. See utils.advect_incident_ti.
    # Amended for the vortex cores ageing on (1-a)*Uhub, the derived form of the march's
    # own U_m, instead of on Uhub. Both isolated cases and T1 are BIT-IDENTICAL -- an
    # isolated rotor's Uhub reads its inflow, not its own wake -- and T2-T5 move only
    # +5e-7 / -3e-6 / -1e-5 / -1e-4. That is the CONTROL: an aligned row has no CVP, so
    # its only cross-flow is the wake's own swirl, which is antisymmetric over the disc
    # and nearly cancels in a rotor average, leaving core size almost irrelevant. Where
    # the term acts, the yawed full-wake cases move 30-1000x as far.
    # Amended with golden_isolated for the cell-centred ground grid: T1 follows the
    # isolated case, T2 is flat at -0.003%, and T3-T5 rise +0.27/+0.56/+0.81% as the
    # ground stops discarding the deficit diffused into it.
    golden_row = [0.99649994412742249, 0.68686537210499177, 0.71923685253416125,
                  0.71950799638754670, 0.71359052339559148]
    sim = _sim([[i * 400.0, 0.0, 0.0] for i in range(5)])
    got_row = [t.Uhub / 8.0 for t in sim.wind_farm.turbines]
    for i, (got, ref) in enumerate(zip(got_row, golden_row)):
        assert abs(got - ref) / ref < 1e-9, (
            f'5D row T{i+1} moved: {got:.17g} vs golden {ref:.17g}')


def test_low_ambient_turbulence_is_refused_not_silently_diverged():
    """Below Du Eq (26)'s zero-crossing the march must RAISE, not return a plausible number.

    Eq (26) carries max(kv1*TI - kv0, 0), exactly zero at TI <= kv0/kv1 = 0.02, i.e.
    I_amb <= 0.0256. With no eddy viscosity nothing damps the central-differenced
    cross-plane advection and the field runs to U ~ -18 m/s while the waked rotor still
    reports Uhub/Uh ~ 0.75.

    The divergence needs CROSS-FLOW, so this yaws the rotor. Unyawed it stays bounded,
    which is why test_wide_farm_marches_without_diverging never caught it -- that test
    asserts the right bound but has never swept I_amb.
    """
    from simulation.utils import DU_KV0, DU_KV1, IU_TO_TI

    i_amb_cliff = (DU_KV0 / DU_KV1) * IU_TO_TI

    def _run(i_amb):
        wf = WindFarmConfig(grid=(1, 2, 6.0, 6.0), D=np.array([80.0]),
                            Zhub=np.array([70.0]), Ct=np.array([0.8]),
                            Cp=np.array([0.47]), yaw=np.array([25.0]),
                            TSR=np.array([7.0]), dist_type='D')
        fc = FieldConfig(Uh=8.0, Zh=70.0, z0=70.0 * 3.3e-4, I_amb=i_amb, min_X=6.0,
                         max_Y=2.0, max_Z=1.5, n_grids=12)
        sim = Simulation(Config(WindFarm=wf, Field=fc, run_prefix='_test_ti'))
        sim.run()
        return sim

    for i_amb in (0.0, 0.02, i_amb_cliff):
        try:
            _run(i_amb)
            raise AssertionError(
                f'I_amb = {i_amb:.4f} gives zero eddy viscosity and must be refused')
        except ValueError:
            pass

    sim = _run(i_amb_cliff * 1.2)   # just above the cliff: must run, and stay bounded
    lo = min(float(np.nanmin(np.asarray(fr.U)))
             for t in sim.wind_farm.turbines for fr in t.wake_field
             if np.asarray(fr.U).size)
    assert lo > -1e-6, f'march went negative just above the cliff: min U = {lo}'


def test_default_ring_survives_the_largest_yaw_we_sweep():
    """Nv and merge_threshold are coupled, and the DEFAULTS have to satisfy that coupling.

    The yawed ring is foreshortened to semi-minor axis R*cos(beta), so its closest
    neighbour spacing is pi*D*cos(beta)/Nv and _merge_close_vortices destroys the ring
    once that falls to merge_threshold*D. wake_solver raises when a RUN violates it, which
    is the right behaviour for a user's config but says nothing about ours: Nv = 49 passed
    every test in this file for months and was invalid from 40 deg up, so 32 of the yaw
    sweep's 117 rows were computed with a ring that collapsed on step one. Nothing noticed,
    because no test and no figure looks at the ring.

    This is pure arithmetic on the defaults -- no simulation -- and it is pinned to the
    sweep's range rather than to the current Nv, so widening the sweep fails here first.
    """
    import simulation.wake_solver  # noqa: F401  (imported for the module-level constants)
    fc = FieldConfig()
    sweep_max_deg = 40.0        # validation_fullwake.run_yaw_sweep tabulates +-40
    spacing = np.pi * np.cos(np.radians(sweep_max_deg)) / fc.Nv
    assert spacing > fc.merge_threshold, (
        f'at the default Nv={fc.Nv} the ring spacing at {sweep_max_deg:.0f} deg is '
        f'{spacing:.4f} D, at or below merge_threshold={fc.merge_threshold:.4f} D: the '
        f'shed ring collapses on the first step over part of the swept yaw range. Lower '
        f'Nv below {np.pi * np.cos(np.radians(sweep_max_deg)) / fc.merge_threshold:.1f}')


def test_nacelle_is_absent_by_default_and_central_when_present():
    """The nacelle disc must be OFF unless a rotor is given one, and act on the centreline.

    The term earns its place by a discrimination test: all three real-rotor datasets
    improve and both actuator-disc datasets degrade. That argument collapses if an
    actuator disc ever silently acquires a nacelle, so the default must be zero and the
    plumbing must carry a per-rotor value without tiling it onto rotors that have none.

    Three things are checked: the default is 0.0 and inert, a mismatched array is
    REJECTED rather than tiled (config._check_consistency), and a real nacelle deepens
    the core while leaving the outer wake alone.
    """
    from config import TurbineConfig
    import simulation.turbine_physics as tp

    assert TurbineConfig.nacelle_D == 0.0, 'a rotor must not get a body by default'

    pos = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
    try:
        WindFarmConfig(pos=pos, D=np.array([126.0]), nacelle_D=np.array([0.0667, 0.0]),
                       dist_type='D')
        raise AssertionError('a length-2 nacelle_D on a 3-rotor farm must raise, not tile')
    except ValueError:
        pass

    wf = WindFarmConfig(pos=pos, D=np.array([126.0]), nacelle_D=np.array([0.05, 0.0, 0.1]),
                        dist_type='D')
    got = [wf[i].nacelle_D for i in range(3)]
    assert got == [0.05, 0.0, 0.1], f'nacelle_D did not reach TurbineConfig per rotor: {got}'

    # a_n = (1 - sqrt(1 - Cd))/2 is full stagnation at the textbook Cd = 1
    a_n = 0.5 * (1.0 - np.sqrt(max(1.0 - tp.NACELLE_CD, 0.0)))
    assert abs(a_n - 0.5) < 1e-12, f'Cd = 1 must give full stagnation, got a_n = {a_n}'

    def _core_and_edge(d_n):
        wf = WindFarmConfig(grid=(1, 1, 6.0, 6.0), D=np.array([0.15]),
                            Zhub=np.array([0.125]), Ct=np.array([0.8]), Cp=np.array([0.47]),
                            yaw=np.array([0.0]), TSR=np.array([7.0]),
                            nacelle_D=np.array([d_n]), dist_type='D')
        fc = FieldConfig(Uh=5.0, Zh=0.125, z0=0.125 * 3.3e-4, I_amb=0.08, min_X=3.0,
                         max_Y=2.0, max_Z=1.5, n_grids=28)
        sim = Simulation(Config(WindFarm=wf, Field=fc, run_prefix='_test_nac'))
        sim.run()
        t = sim.wind_farm.turbines[0]
        fr = t.wake_field[min(3, len(t.wake_field) - 1)]
        y, z = np.asarray(t.yloc), np.asarray(t.zloc)
        U = np.asarray(fr.U) / float(t.Uhub)
        core = (np.abs(y) <= 0.10 * t.D) & (np.abs(z - t.Zhub) <= 0.10 * t.D)
        edge = (np.abs(y) >= 0.30 * t.D) & (np.abs(y) <= 0.45 * t.D) \
            & (np.abs(z - t.Zhub) <= 0.10 * t.D)
        return float(U[core].mean()), float(U[edge].mean())

    off_core, off_edge = _core_and_edge(0.0)
    on_core, on_edge = _core_and_edge(10.0 / 150.0)
    assert on_core < off_core - 1e-6, (
        f'a nacelle must deepen the core: {on_core:.6f} vs {off_core:.6f}')
    assert abs(on_edge - off_edge) < abs(off_core - on_core), (
        f'the nacelle must act centrally: edge moved {abs(on_edge-off_edge):.6f}, '
        f'core moved {abs(off_core-on_core):.6f}')


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted((k, v) for k, v in list(globals().items())
                           if k.startswith('test_') and callable(v)):
        try:
            fn()
            print(f'PASS  {name}')
        except AssertionError as e:
            failures += 1
            print(f'FAIL  {name}\n      {e}')
    print(f'\n{"all invariants hold" if not failures else f"{failures} FAILED"}')
    sys.exit(1 if failures else 0)
