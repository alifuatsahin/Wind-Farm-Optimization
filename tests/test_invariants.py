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


def test_identical_rotors_see_identical_inflow():
    """In a 2x2 farm the two downstream rotors are geometrically identical."""
    d = _row([[0, 0, 0], [0, 560, 0], [560, 0, 0], [560, 560, 0]])
    a, b = d[(7, 0)], d[(7, 7)]
    assert abs(a - b) / a < 0.01, f'identical rotors disagree: {a:.6f} vs {b:.6f}'
    assert a < 0.9, f'downstream rotor is not waked at all ({a:.6f}): wake chain broken'

def test_isolated_yawed_turbine_loses_exactly_the_cosine_power():
    """An isolated turbine's efficiency must be exactly Cp(beta)/Cp(0) = cos^1.88(beta).

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
        ref = np.cos(np.deg2rad(beta)) ** 1.88
        assert abs(eta - ref) / ref < 1e-12, (
            f'beta={beta}: eta={eta:.10f} but Cp ratio is {ref:.10f}; the power disc and '
            f'the reference disc disagree')

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
