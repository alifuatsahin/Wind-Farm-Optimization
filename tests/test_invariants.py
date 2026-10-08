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
