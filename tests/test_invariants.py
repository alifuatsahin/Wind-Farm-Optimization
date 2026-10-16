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


def _row(pos, max_Y=3.0):
    """A farm at explicit absolute positions, in metres."""
    n = len(pos)
    wf = WindFarmConfig(pos=np.array(pos, float), D=np.array([80.0]),
                        Zhub=np.array([70.0]), Ct=np.array([0.8]), Cp=np.array([0.47]),
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

    The bound below is deliberately loose enough for the width floor's 1.74%; it exists to
    catch the 12% class, not to certify the 1.74%. Getting to <1% needs the turbulence age
    transported as a field rather than read off rotor positions.
    """
    base = _row([[400, 0, 0], [800, 0, 0]])[(10, 0)]
    with_a = _row([[0, 560, 0], [400, 0, 0], [800, 0, 0]])[(10, 0)]
    err = abs(with_a - base) / base
    assert err < 0.02, (
        f'a rotor 7D to the side and 5D upstream moved the column by {100*err:.2f}% '
        f'({base:.6f} -> {with_a:.6f}): the budget length scale is reading rotor positions '
        f'rather than the local flow')


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
    # NUT_CRESPO's value here is 0.7574209144261419, and under it the 1% assertion above
    # FAILS at 9.3%. The discrepancy is
    # OWNED BY THE CLOSURE, not by the march: NuT_model measures x from each UPSTREAM rotor,
    # so its nu_T depends on which rotors the station grouping put upstream. Both k_w-based
    # closures read nu_T from local fields instead and are grouping-invariant -- measured
    # 0.267%, and this test passes under them. Three points isolate it: Crespo with the old
    # non-conservative operator 9.3%, Crespo with the conservative operator 9.3%, Du Eq (26)
    # with the conservative operator 0.267%. The diffusion operator is NOT the cause.
    assert abs(grouped - 0.72433053902390099) / 0.72433053902390099 < 1e-9, (
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
    golden_isolated = {0.0: 0.99693986652197308, 25.0: 0.99761049965016602}
    for beta, ref in golden_isolated.items():
        got = _sim([[0.0, 0.0, 0.0]], [beta]).wind_farm.turbines[0].Uhub / 8.0
        assert abs(got - ref) / ref < 1e-9, (
            f'isolated turbine at yaw={beta} moved: {got:.17g} vs golden {ref:.17g}')

    # These are the GAMMA_TRAILING_K = 0.45 values. They were briefly re-recorded for
    # k = 1.0 (T2 moved 0.006%, T5 0.086%, T1 stayed bit-identical) and then restored when
    # k went back to Zong's measured 0.45 -- see GAMMA_TRAILING_K's own note. That episode
    # is why this test records numbers rather than tolerating drift: it is the only thing
    # that would have caught the constant changing underneath the row cases.
    # These are the NUT_FROM_TKE values. Under NUT_CRESPO the row is
    # [.., 0.68561188238556825, 0.70839363258543719, 0.69427784837223705, 0.68659866597497632],
    # which is operator-independent because Crespo's nu_T
    # varies across the plane only at |dy| = 3D, where the deficit gradient is zero, so
    # div(nu_T grad Uw) and nu_T*lap(Uw) agree to the last bit. Measured identical on all
    # three datasets. Under the k_w closures the row becomes
    # Re-record deliberately if the closure default changes; do not relax. This row RISES from
    # T2 to T4 and then flattens (0.651 / 0.666 / 0.690 / 0.681), which is the deep-array
    # plateau. BUDGET_AGE = 'width' gave a monotonic 0.651 / 0.606 / 0.601 / 0.602; the
    # disqualified 'farm' reference gave 0.651 / 0.673 / 0.691 / 0.681, i.e. the transported
    # age recovers it without reading rotor positions.
    golden_row = [0.99693986652197308, 0.65147963814922771, 0.66624274492671842,
                  0.69037240002761913, 0.68096794287884532]
    sim = _sim([[i * 400.0, 0.0, 0.0] for i in range(5)])
    got_row = [t.Uhub / 8.0 for t in sim.wind_farm.turbines]
    for i, (got, ref) in enumerate(zip(got_row, golden_row)):
        assert abs(got - ref) / ref < 1e-9, (
            f'5D row T{i+1} moved: {got:.17g} vs golden {ref:.17g}')


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
