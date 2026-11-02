"""Checks run_sweep.sh makes before spending hours on the optimiser-efficiency sweep."""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

REFERENCE = {3: 0.576264384282629, 8: 0.489963688862883}

TOL = 1e-5


def _objective(n_t, n_grids=28):
    import make_paper_figures as M
    from simulation import Simulation
    from copy import deepcopy

    cfg = M._fullwake_column_cfg(n_t, n_grids=n_grids)
    cfg.WindFarm.yaw = np.linspace(20.0, 0.0, n_t)
    sim = Simulation(deepcopy(cfg))
    sim.run()
    return sim.calculate_objective()


def verify():
    import jax

    print("  backend %s, x64=%s" % (jax.default_backend(), jax.config.jax_enable_x64))
    if jax.default_backend() != "cpu":
        print("  note: compare against 'highest' matmul precision if this fails")
    ok = True
    for n_t, want in sorted(REFERENCE.items()):
        t0 = time.perf_counter()
        got = _objective(n_t)
        rel = abs(got - want) / abs(want)
        flag = "ok" if rel <= TOL else "MISMATCH"
        if rel > TOL:
            ok = False
        print("  N_t=%d  got %.15f  want %.15f  rel %.2e  %s  (%.1f s)"
              % (n_t, got, want, rel, flag, time.perf_counter() - t0))
    if not ok:
        print("\n  Reference values come from CPU float32. If this is a GPU and the\n"
              "  mismatch is near 1e-3, it is most likely TF32 matmuls: re-run with\n"
              "  JAX_DEFAULT_MATMUL_PRECISION=highest and compare again.")
    return ok


def benchmark(nt_list, seeds_b, budget, evals_per_dim, seeds):
    """Extrapolate the sweep from one timed evaluation at each end of the N_t range."""
    import jax

    lo, hi = min(nt_list), max(nt_list)
    timed = {}
    for n_t in sorted({lo, hi}):
        _objective(n_t)
        t0 = time.perf_counter()
        _objective(n_t)
        timed[n_t] = time.perf_counter() - t0
        print("  N_t=%-2d  %6.2f s per evaluation" % (n_t, timed[n_t]))

    def secs(n_t):
        if n_t in timed:
            return timed[n_t]
        if lo == hi:
            return timed[lo]
        w = (n_t - lo) / float(hi - lo)
        return timed[lo] + w * (timed[hi] - timed[lo])

    total = 0.0
    n_evals = 0
    print("\n  %-5s %8s %10s %10s" % ("N_t", "budget", "evals", "hours"))
    for n_t in nt_list:
        bud = max(budget, evals_per_dim * n_t)
        e = 3 * seeds_b * bud + 2 * (bud // 3)
        t = e * secs(n_t)
        n_evals += e
        total += t
        print("  %-5d %8d %10d %10.2f" % (n_t, bud, e, t / 3600.0))

    a_evals = 3 * seeds * budget + 2 * (budget // 3)
    print("\n  panel (b) total: %d evaluations, %.1f hours" % (n_evals, total / 3600.0))
    print("  panel (a) would add %d evaluations (%.1f h) if not already cached"
          % (a_evals, a_evals * secs(3) / 3600.0))
    print("  backend %s" % jax.default_backend())
    cached = os.path.join("runs", "paper_figures", "bo_efficiency_runs.json")
    if os.path.exists(cached):
        import json
        with open(cached) as fh:
            print("  %d curves already cached in %s (these will be skipped)"
                  % (len(json.load(fh)), cached))
    else:
        print("  no cache at %s -- nothing to resume from" % cached)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--nt", default="2 3 4 6 8")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--seeds-b", type=int, default=5, dest="seeds_b")
    p.add_argument("--budget", type=int, default=120)
    p.add_argument("--evals-per-dim", type=int, default=50, dest="evals_per_dim")
    a = p.parse_args()

    if a.verify and not verify():
        return 1
    if a.benchmark:
        benchmark([int(v) for v in a.nt.split()], a.seeds_b, a.budget,
                  a.evals_per_dim, a.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
