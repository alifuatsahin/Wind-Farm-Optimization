"""Reference optimisers for the TuRBO comparison in section ssec:bo_efficiency."""
import json
import time
from typing import Callable, List, Optional

import numpy as np


def _run(evaluate, x, history, log_path=None):
    """Evaluate one point and append it in `Optimizer.history`'s format."""
    t0 = time.perf_counter()
    value = float(evaluate(np.asarray(x, float)))
    rec = {
        "eval": len(history),
        "value": value,
        "best_so_far": float(max([value] + [r["value"] for r in history])),
        "params": [float(v) for v in np.asarray(x, float)],
        "seconds": float(time.perf_counter() - t0),
    }
    history.append(rec)
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    return value


def random_search(evaluate: Callable, bounds: np.ndarray, max_evals: int,
                  seed: int = 0, log_path: Optional[str] = None) -> List[dict]:
    """Uniform sampling of the box."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(bounds, float).T
    history: List[dict] = []
    while len(history) < max_evals:
        _run(evaluate, rng.uniform(lo, hi), history, log_path)
    return history


def genetic_algorithm(evaluate: Callable, bounds: np.ndarray, max_evals: int,
                      seed: int = 0, pop_size: Optional[int] = None, n_elite: int = 2,
                      p_crossover: float = 0.9, p_mutate: float = 0.2,
                      blx_alpha: float = 0.5, log_path: Optional[str] = None,
                      sigma_frac: float = 0.1) -> List[dict]:
    """Real-coded GA: tournament selection, BLX-alpha crossover, Gaussian mutation."""
    rng = np.random.default_rng(seed)
    b = np.asarray(bounds, float)
    lo, hi = b[:, 0], b[:, 1]
    dim = len(lo)
    if pop_size is None:
        pop_size = max(10, 2 * dim)
    n_elite = min(n_elite, pop_size - 1)
    sigma = sigma_frac * (hi - lo)
    history: List[dict] = []

    pop = rng.uniform(lo, hi, size=(pop_size, dim))
    fit = np.array([_run(evaluate, x, history, log_path) for x in pop])

    while len(history) < max_evals:
        order = np.argsort(-fit)
        new = [pop[i].copy() for i in order[:n_elite]]
        while len(new) < pop_size:
            pa, pb = (max(rng.integers(0, pop_size, 2), key=lambda i: fit[i])
                      for _ in range(2))
            c = pop[pa].copy()
            if rng.random() < p_crossover:
                x1, x2 = pop[pa], pop[pb]
                d = np.abs(x1 - x2)
                c = rng.uniform(np.minimum(x1, x2) - blx_alpha * d,
                                np.maximum(x1, x2) + blx_alpha * d)
            if rng.random() < p_mutate:
                c = c + rng.normal(0.0, sigma)
            new.append(np.clip(c, lo, hi))

        pop = np.array(new)
        fit_new = []
        for i, x in enumerate(pop):
            if len(history) >= max_evals:
                fit_new.append(-np.inf)
                continue
            fit_new.append(fit[order[i]] if i < n_elite
                           else _run(evaluate, x, history, log_path))
        fit = np.array(fit_new)
    return history


def zong_gradient_ascent(evaluate: Callable, bounds: np.ndarray, max_evals: int,
                         beta_max: float = 25.0, sign: float = 1.0,
                         step0: float = 5.0, step_min: float = 0.1,
                         fd_step: float = 1.0, heuristic_start: bool = True,
                         log_path: Optional[str] = None) -> List[dict]:
    """Zong & Porte-Agel (2021) section 4.3, reproduced rather than reinvented."""
    b = np.asarray(bounds, float)
    lo, hi = b[:, 0], b[:, 1]
    dim = len(lo)
    history: List[dict] = []

    i = np.arange(dim, 0, -1) - 1.0
    x = (np.clip(sign * beta_max * i / max(dim, 1), lo, hi) if heuristic_start
         else np.zeros(dim))
    f = _run(evaluate, x, history, log_path)

    step = step0
    while step >= step_min and len(history) < max_evals:
        g = np.zeros(dim)
        for k in range(dim):
            if len(history) >= max_evals:
                return history
            xp = x.copy()
            xp[k] = np.clip(xp[k] + fd_step, lo[k], hi[k])
            h = xp[k] - x[k]
            g[k] = 0.0 if h == 0 else (_run(evaluate, xp, history, log_path) - f) / h
        n = np.linalg.norm(g)
        if n == 0.0 or len(history) >= max_evals:
            break
        x_try = np.clip(x + step * g / n, lo, hi)
        f_try = _run(evaluate, x_try, history, log_path)
        if f_try > f:
            x, f = x_try, f_try
        else:
            step *= 0.5
    return history


def evals_to_within(history: List[dict], target: float, tol: float = 1e-3):
    """Evaluations to come within `tol` (relative) of `target`; None if never."""
    if not history:
        return None
    best = np.maximum.accumulate([r["value"] for r in history])
    hit = np.flatnonzero(best >= target * (1.0 - tol))
    return int(hit[0]) + 1 if hit.size else None
