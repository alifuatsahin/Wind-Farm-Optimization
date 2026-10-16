"""Validation against the three-turbine full-wake experiment in data/.

A DIFFERENT experimental run from the eight-turbine row in data/exp_data.h5, so none of
that case's calibration carries over -- in particular its I_amb = 0.055 does not apply here.

What the dataset provides, and what each part is good for:

  fullwake_contour_avg_exp.h5   U/Uinf on a hub-height plane, 634 x 177, x/D = -0.08..14.99,
                                y/D = +-2.1, for three cases A/B/C
  fullwake_contour_std_exp.h5   turbulence intensity on the same grid
  fullwake_rpm_power_exp_full.csv  117 yaw combinations with MEASURED rpm and power per
                                rotor. The paper's power comparison was dropped because
                                neither earlier dataset measured power; this one does.

Geometry and conditions are read off the data rather than assumed:

  * rotors at x/D = 0, 5, 10 -- from the centreline deficit, which drops to 0.44 after T1,
    0.36 after T2 and 0.22 after T3, with the rotor planes masked out.
  * I_amb = 0.070 -- measured directly in the band |y/D| > 1.5, where U/Uinf = 0.999
    confirms the band is undisturbed. 0.071 at x/D = 0.5 settling to 0.068 by 4D. This is a
    MEASURED ambient, not a swept one, which is what makes this dataset worth having.
  * case C is the UNYAWED baseline. Its wake centre sits within 0.03D of y = 0 at every
    clean station, while A and B are a mirror pair at about +-0.3D. The three row-groups in
    Haohua-fullwake.csv are ordered baseline-first, NOT alphabetically by case, so matching
    cases to yaw angles through that file gets A and C backwards.

Still unresolved: the yaw magnitudes for cases A and B. The figures label panels only
(a)/(b)/(c) and carry no angle annotation, so the pair cannot be identified from the files
alone. `candidate_yaw_sets` holds the two mirror pairs that best match the measured powers;
resolve_ab() runs both and reports which assignment reproduces the measured deflection.
"""
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt

from config import Config, FieldConfig, WindFarmConfig
from simulation import Simulation
from simulation.viz_style import style_figure, style_axes
from validation import (extract_farm_hub_height_slice, compute_error_metrics,
                        plot_contour_comparison, plot_profile_comparison, _panel_label)

AVG_PATH = 'data/fullwake_contour_avg_exp.h5'
STD_PATH = 'data/fullwake_contour_std_exp.h5'
POWER_PATH = 'data/fullwake_rpm_power_exp_full.csv'

#: WIRE-01, CONFIRMED against the source paper: Zong & Porte-Agel (2021), Renewable Energy
#: 170, 1228-1244, doi:10.1016/j.renene.2021.02.059, which is where this dataset comes from.
#: Sec 2 gives D = 0.15 m, Z_h = 0.125 m, S_x = 5D, U_h = 4.9 m/s and hub-height turbulence
#: 7.1% (we measure 7.0% from the std file). Sec 4.2 gives C_t0 = 0.82 and C_p0 = 0.31.
#:
#: C_p0 was 0.35 here and in validation._base_case with no source. It cancels from every
#: power RATIO and never enters the velocity field, so no metric in this module moved when
#: it was corrected -- but it was wrong for absolute power. U_h was likewise 4.314 m/s,
#: carried over from the EIGHT-turbine run, which is a different experiment.
D = 0.15
ZHUB = 0.125
CT = 0.82
CP = 0.31
TSR = 4.5
UH = 4.9

#: Measured in-dataset; see the module docstring.
I_AMB = 0.070

#: z0 is not measured for this tunnel. Hold log(Zh/z0) at the model's full-scale default,
#: exactly as _base_case does, so the inflow profile SHAPE is unchanged and z0 does not
#: become a second free parameter.
Z0 = 0.03 * (ZHUB / 90.0)

POSITIONS_D = (0.0, 5.0, 10.0)

#: Baseline row of the measured power table, (0, 0, 0).
BASELINE_YAW = (0.0, 0.0, 0.0)

#: The two mirror pairs that best match the measured normalised powers. Which is A and
#: which is B is decided by deflection sign, not by these numbers; see resolve_ab().
candidate_yaw_sets = {'pos': (25.0, 15.0, 0.0), 'neg': (-25.0, -20.0, 0.0)}


def load_fullwake_case(case, kind='avg'):
    """x/D, y/D and the measured field for one case. `z` is U/Uinf for kind='avg' and
    turbulence intensity for kind='std'; both carry ~2% scattered NaN dropouts."""
    path = AVG_PATH if kind == 'avg' else STD_PATH
    with h5py.File(path, 'r') as h:
        g = h['case_%s' % case.upper()]
        return dict(x_D=g['x'][0, :].copy(), y_D=g['y'][:, 0].copy(), z=g['z'][:].copy())


def measured_power(yaw):
    """Measured per-rotor power for one yaw combination, or None if it is not tabulated."""
    t = np.loadtxt(POWER_PATH, delimiter=',')
    hit = t[(t[:, 0] == yaw[0]) & (t[:, 1] == yaw[1]) & (t[:, 2] == yaw[2])]
    return None if len(hit) == 0 else hit[0, 6:9]


def model_power(wind_farm):
    """Per-rotor power in the model's own terms: P ~ Cp(beta) * Uhub^3.

    params.Cp already carries the cos^CP_YAW_EXP yaw factor, so this is the same quantity the
    objective maximises -- not the hub-height line proxy compute_power_metrics builds for
    datasets that do not measure power. Only ratios are compared, so the omitted
    0.5*rho*A cancels.
    """
    return np.array([t.Cp * t.Uhub ** 3 for t in wind_farm.turbines])


def run_fullwake_case(yaw, i_amb=I_AMB, n_grids=20, x_D_max=15.0, y_D_halfwidth=2.1,
                      **field_kw):
    """Three-turbine row at the measured spacing. Deliberately does NOT pass
    vortex_nu='abl': that form is built on u* and z0 and belongs to the atmospheric cases,
    while this is a wind tunnel, so the default (Squire) applies as for the other rig.

    `field_kw` goes straight to FieldConfig, for sweeping things like merge_threshold."""
    wf = WindFarmConfig(
        pos=np.array([[xd * D, 0.0, 0.0] for xd in POSITIONS_D], dtype=float),
        D=np.array([D]), Zhub=np.array([ZHUB]), Ct=np.array([CT]), Cp=np.array([CP]),
        yaw=np.array(yaw, dtype=float), TSR=np.array([TSR]), dist_type='m',
    )
    field = FieldConfig(
        Uh=UH, Zh=ZHUB, I_amb=i_amb, z0=Z0,
        min_X=(x_D_max - max(POSITIONS_D)) * 1.1,
        max_Y=y_D_halfwidth * 2 * 1.05,
        max_Z=2.0, n_grids=n_grids, **field_kw,
    )
    sim = Simulation(Config(WindFarm=wf, Field=field, run_prefix='validate_fullwake'))
    sim.run()
    return sim


def wake_centre(y_D, z, valid_mask=None):
    """Deficit-weighted wake centre, in D. Weighted by (1-U)^2 so the core dominates and
    the noisy wake edges do not drag the centroid outwards."""
    out = np.full(z.shape[1], np.nan)
    inner = np.abs(y_D) < 1.5
    for i in range(z.shape[1]):
        col = z[:, i]
        m = inner & np.isfinite(col)
        if valid_mask is not None:
            m = m & valid_mask
        d = np.clip(1.0 - col[m], 0.0, None) ** 2
        if d.sum() > 0:
            out[i] = float(np.sum(y_D[m] * d) / np.sum(d))
    return out


def plot_power_comparison(cases, save_path=None):
    """Measured vs modelled per-rotor power, normalised to T1 of the same case.

    Normalising within the case removes the unknown absolute scale of the measured power
    (units are not stated in the file) and the model's omitted 0.5*rho*A together, so the
    comparison is of the wake-induced power distribution, which is what is actually being
    tested.
    """
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    style_figure(fig)
    n = len(cases)
    w = 0.8 / (2 * n)
    xs = np.arange(3)
    for k, (label, meas, mod) in enumerate(cases):
        ax.bar(xs + (2 * k) * w - 0.4 + w / 2, meas / meas[0], w,
               label='%s, measured' % label, alpha=0.85)
        ax.bar(xs + (2 * k + 1) * w - 0.4 + w / 2, mod / mod[0], w,
               label='%s, model' % label, alpha=0.85, hatch='///')
    ax.set_xticks(xs)
    ax.set_xticklabels(['T1', 'T2', 'T3'])
    ax.set_ylabel(r'$P_i / P_1$')
    ax.legend(fontsize=6, frameon=False, ncol=n)
    style_axes(ax)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_deflection_comparison(x_D, yc_exp, yc_sim, save_path=None, label=''):
    """Measured vs modelled wake centreline. The rotor planes are masked in the PIV, which
    shows up as a spike in the measured centre on either side of x/D = 5 and 10; those
    columns are dropped rather than compared."""
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    style_figure(fig)
    ax.plot(x_D, yc_exp, lw=1.2, label='Experiment')
    ax.plot(x_D, yc_sim, lw=1.2, ls='--', label='Model')
    for xr in POSITIONS_D[1:]:
        ax.axvline(xr, color='0.7', lw=0.6, zorder=0)
    ax.axhline(0.0, color='0.7', lw=0.6, zorder=0)
    ax.set_xlabel(r'$x/D$')
    ax.set_ylabel(r'$y_c/D$')
    if label:
        ax.set_title(label, fontsize=7)
    ax.legend(fontsize=6, frameon=False)
    style_axes(ax)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def _near_rotor(x_D, halfwidth=0.4):
    """Columns within `halfwidth` D of a rotor plane, where the PIV is masked/blocked."""
    bad = np.zeros_like(x_D, dtype=bool)
    for xr in POSITIONS_D:
        bad |= np.abs(x_D - xr) < halfwidth
    return bad


def validate_fullwake_case(case, yaw, save_dir='runs/validation_fullwake', i_amb=I_AMB,
                           x_stride=4, n_grids=20, make_plots=True, tag=None):
    """Field metrics, deflection and power for one case."""
    tag = tag or case
    exp = load_fullwake_case(case, 'avg')
    sim = run_fullwake_case(yaw, i_amb=i_amb, n_grids=n_grids,
                            x_D_max=float(np.nanmax(exp['x_D'])),
                            y_D_halfwidth=float(np.nanmax(np.abs(exp['y_D']))))

    x_D = exp['x_D'][::x_stride]
    y_D = exp['y_D']
    U_exp = exp['z'][:, ::x_stride] * UH          # data is normalised; restore scale
    slice_ = extract_farm_hub_height_slice(sim.wind_farm, x_D * D, y_D * D)
    U_sim = slice_['U']

    metrics = compute_error_metrics(U_sim, U_exp, slice_['Uh'], UH)
    metrics['case'] = case
    metrics['yaw'] = tuple(float(v) for v in yaw)

    keep = ~_near_rotor(x_D)
    yc_exp = wake_centre(y_D, U_exp / UH)
    yc_sim = wake_centre(y_D, U_sim / slice_['Uh'])
    both = keep & np.isfinite(yc_exp) & np.isfinite(yc_sim)
    metrics['deflection_rmse_D'] = float(np.sqrt(np.mean((yc_sim[both] - yc_exp[both]) ** 2)))
    metrics['deflection_exp_mean_D'] = float(np.mean(yc_exp[both]))
    metrics['deflection_sim_mean_D'] = float(np.mean(yc_sim[both]))

    meas = measured_power(yaw)
    mod = model_power(sim.wind_farm)
    metrics['power_model_ratio'] = (mod / mod[0]).tolist()
    if meas is not None:
        metrics['power_meas_ratio'] = (meas / meas[0]).tolist()
        metrics['power_ratio_err'] = float(
            np.max(np.abs(mod / mod[0] - meas / meas[0])))

    if make_plots:
        os.makedirs(save_dir, exist_ok=True)
        plot_contour_comparison(x_D, y_D, U_exp, U_sim, UH, slice_['Uh'],
                                save_path=os.path.join(save_dir, 'contour_%s.png' % tag))
        plot_profile_comparison(x_D, y_D, U_exp, U_sim, UH, slice_['Uh'],
                                stations=[2, 3, 7, 8, 12, 14],
                                save_path=os.path.join(save_dir, 'profiles_%s.png' % tag))
        plot_deflection_comparison(x_D[keep], yc_exp[keep], yc_sim[keep],
                                   save_path=os.path.join(save_dir, 'deflection_%s.png' % tag),
                                   label='case %s, yaw = %s' % (case, tuple(yaw)))
    return metrics, sim


def _report(m):
    print('  case %-2s yaw=%-18s RMSE(U/Uh)=%.4f  MAE=%.4f  bias=%+.4f  R2=%.4f  N=%d'
          % (m['case'], str(m['yaw']), m['rmse'], m['mae'], m['bias'], m['r2'],
             m['n_points']))
    print('          deflection: exp %+.3f D, model %+.3f D, RMSE %.3f D'
          % (m['deflection_exp_mean_D'], m['deflection_sim_mean_D'],
             m['deflection_rmse_D']))
    if 'power_meas_ratio' in m:
        print('          P/P1 measured %s' % np.round(m['power_meas_ratio'], 4))
        print('          P/P1 model    %s   worst %+.1f%%'
              % (np.round(m['power_model_ratio'], 4), 100 * m['power_ratio_err']))
    else:
        print('          P/P1 model    %s   (yaw not in the measured table)'
              % np.round(m['power_model_ratio'], 4))


def validate_baseline(save_dir='runs/validation_fullwake', **kw):
    """Case C: the unyawed baseline, the only case whose yaw is settled by the data."""
    print('Full-wake experiment, 3 turbines at 5D, I_amb = %.3f (measured in-dataset)'
          % kw.get('i_amb', I_AMB))
    m, sim = validate_fullwake_case('C', BASELINE_YAW, save_dir=save_dir, **kw)
    _report(m)
    return m, sim


def resolve_ab(save_dir='runs/validation_fullwake', x_stride=4, **kw):
    """Which of the candidate yaw sets is case A and which is case B.

    Scores all FOUR pairings on the deflection profile rather than comparing means, so a
    case whose mean happens to match but whose streamwise shape does not cannot win. The
    rotor planes are dropped: the PIV masks them, which spikes the measured centroid.

    The discriminator is the model's own yaw convention acting on measured data, so the
    assignment is read off rather than assumed -- which is the point, since matching the
    cases through Haohua-fullwake.csv gets A and C backwards.
    """
    ref = load_fullwake_case('A', 'avg')
    x_D = ref['x_D'][::x_stride]
    y_D = ref['y_D']
    keep = ~_near_rotor(x_D)

    meas = {}
    for case in ('A', 'B'):
        e = load_fullwake_case(case, 'avg')
        meas[case] = wake_centre(y_D, e['z'][:, ::x_stride] / 1.0)

    model = {}
    for key, yaw in candidate_yaw_sets.items():
        sim = run_fullwake_case(yaw, i_amb=kw.get('i_amb', I_AMB),
                                n_grids=kw.get('n_grids', 20),
                                x_D_max=float(np.nanmax(ref['x_D'])),
                                y_D_halfwidth=float(np.nanmax(np.abs(y_D))))
        sl = extract_farm_hub_height_slice(sim.wind_farm, x_D * D, y_D * D)
        model[key] = (wake_centre(y_D, sl['U'] / sl['Uh']), yaw, sim)

    print('  measured mean deflection:  A %+.3f D   B %+.3f D'
          % (np.nanmean(meas['A'][keep]), np.nanmean(meas['B'][keep])))
    for key, (yc, yaw, _) in model.items():
        print('  model yaw %-18s mean deflection %+.3f D'
              % (str(yaw), np.nanmean(yc[keep])))

    print('  deflection RMSE over %d stations, all four pairings:' % keep.sum())
    score = {}
    for case in ('A', 'B'):
        for key, (yc, yaw, _) in model.items():
            both = keep & np.isfinite(yc) & np.isfinite(meas[case])
            r = float(np.sqrt(np.mean((yc[both] - meas[case][both]) ** 2)))
            score[(case, key)] = r
            print('    case %s  vs  yaw %-18s  RMSE %.4f D' % (case, str(yaw), r))

    best = min([(score[('A', k1)] + score[('B', k2)], k1, k2)
                for k1 in model for k2 in model if k1 != k2])
    print('  -> best assignment: A = %s, B = %s  (summed RMSE %.4f D)'
          % (candidate_yaw_sets[best[1]], candidate_yaw_sets[best[2]], best[0]))
    return dict(score=score, assignment={'A': best[1], 'B': best[2]}, model=model,
                meas=meas, x_D=x_D, keep=keep)


if __name__ == '__main__':
    validate_baseline()


# --------------------------------------------------------------------------------------
# The 117-point yaw sweep. This is the objective surface the optimiser actually searches,
# and the only test that measures the yaw-sign asymmetry across the whole (g1, g2) plane
# rather than at one or two angles.
#
# n_grids = 28: the asymmetry is NOT converged below that (+0.847 / +0.628 / +0.465% at
# n_grids 14 / 20 / 28) but flattens above it (+0.465 / +0.452 / +0.498% at 28 / 36 / 44).
# Quoting an asymmetry from a coarser grid overstates it.
# --------------------------------------------------------------------------------------

def run_yaw_sweep(n_grids=28, out='runs/validation_fullwake/sweep.npz', limit=None):
    """March every tabulated yaw combination and store per-rotor model power."""
    tab = np.loadtxt(POWER_PATH, delimiter=',')
    if limit:
        tab = tab[:limit]
    rows = []
    for i, r in enumerate(tab):
        yaw = (float(r[0]), float(r[1]), float(r[2]))
        sim = run_fullwake_case(yaw, n_grids=n_grids)
        rows.append(np.concatenate([r[:3], model_power(sim.wind_farm), r[6:10]]))
        if (i + 1) % 10 == 0 or i + 1 == len(tab):
            print('  %3d/%d  yaw=%-20s' % (i + 1, len(tab), str(yaw)), flush=True)
    arr = np.array(rows)   # [g1,g2,g3, Pm1,Pm2,Pm3, Pe1,Pe2,Pe3, Petot]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, sweep=arr, n_grids=n_grids)
    return arr


def _surface(ax, g1, g2, v, title, vmin=None, vmax=None, cmap='turbo'):
    u1, u2 = np.unique(g1), np.unique(g2)
    M = np.full((len(u2), len(u1)), np.nan)
    for a, b, c in zip(g1, g2, v):
        M[np.searchsorted(u2, b), np.searchsorted(u1, a)] = c
    im = ax.pcolormesh(u1, u2, M, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
    ax.set_xlabel(r'$\gamma_1$ (deg)')
    ax.set_ylabel(r'$\gamma_2$ (deg)')
    ax.set_title(title, fontsize=7)
    style_axes(ax)
    return im


def report_yaw_sweep(path='runs/validation_fullwake/sweep.npz',
                     save_dir='runs/validation_fullwake'):
    d = np.load(path)
    a = d['sweep']
    g1, g2 = a[:, 0], a[:, 1]
    Pm = a[:, 3:6].sum(axis=1)
    Pe = a[:, 9]
    base = (g1 == 0) & (g2 == 0) & (a[:, 2] == 0)
    pm, pe = Pm / Pm[base][0], Pe / Pe[base][0]

    diff = pm - pe
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((pe - pe.mean()) ** 2))
    print('Total power, normalised to the (0,0,0) baseline, over %d yaw combinations'
          % len(a))
    print('  RMSE=%.4f  MAE=%.4f  bias=%+.4f  R2=%.4f'
          % (np.sqrt(np.mean(diff ** 2)), np.mean(np.abs(diff)), diff.mean(),
             1 - ss_res / ss_tot))

    ko, km = int(np.argmax(pe)), int(np.argmax(pm))
    print('  measured optimum: yaw=(%+.0f,%+.0f,0)  P/P0=%.4f' % (g1[ko], g2[ko], pe[ko]))
    print('  model    optimum: yaw=(%+.0f,%+.0f,0)  P/P0=%.4f' % (g1[km], g2[km], pm[km]))
    print('  model P/P0 AT the measured optimum: %.4f   (gap to its own optimum %.4f)'
          % (pm[ko], pm[km] - pm[ko]))
    print('  measured P/P0 at the model optimum: %.4f   -> LOST GAIN %.4f of baseline'
          % (pe[km], pe[ko] - pe[km]))

    key = {(x, y): i for i, (x, y) in enumerate(zip(g1, g2))}
    pairs = [(i, key[(-x, -y)]) for i, (x, y) in enumerate(zip(g1, g2))
             if (-x, -y) in key and (x > 0 or (x == 0 and y > 0))]
    am = np.array([200 * (pm[i] - pm[j]) / (pm[i] + pm[j]) for i, j in pairs])
    ae = np.array([200 * (pe[i] - pe[j]) / (pe[i] + pe[j]) for i, j in pairs])
    print('  mirrored pairs: %d   measured asym mean %+.2f%%   model %+.2f%%   captured %.0f%%'
          % (len(pairs), ae.mean(), am.mean(), 100 * am.mean() / ae.mean()))
    print('  pairs where the model has the WRONG SIGN: %d of %d'
          % (int(np.sum(np.sign(am) != np.sign(ae))), len(pairs)))

    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
    style_figure(fig)
    lo, hi = min(pe.min(), pm.min()), max(pe.max(), pm.max())
    im = _surface(axes[0], g1, g2, pe, 'Measured $P/P_0$', lo, hi)
    fig.colorbar(im, ax=axes[0])
    im = _surface(axes[1], g1, g2, pm, 'Model $P/P_0$', lo, hi)
    fig.colorbar(im, ax=axes[1])
    im = _surface(axes[2], g1, g2, pm - pe, 'Model $-$ measured', -0.12, 0.12, 'RdBu_r')
    fig.colorbar(im, ax=axes[2])
    for ax, L in zip(axes, 'abc'):
        _panel_label(ax, L)
    axes[0].plot(g1[ko], g2[ko], 'w*', ms=11, mec='k')
    axes[1].plot(g1[km], g2[km], 'w*', ms=11, mec='k')
    fig.savefig(os.path.join(save_dir, 'sweep_surface.png'), dpi=200, bbox_inches='tight')

    fig2, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
    style_figure(fig2)
    ax[0].plot([lo, hi], [lo, hi], color='0.6', lw=0.8)
    ax[0].scatter(pe, pm, s=9, alpha=0.75)
    ax[0].set_xlabel('Measured $P/P_0$')
    ax[0].set_ylabel('Model $P/P_0$')
    _panel_label(ax[0], 'a')
    style_axes(ax[0])
    m = max(abs(ae).max(), abs(am).max()) * 1.1
    ax[1].plot([-m, m], [-m, m], color='0.6', lw=0.8)
    ax[1].axhline(0, color='0.85', lw=0.6)
    ax[1].axvline(0, color='0.85', lw=0.6)
    ax[1].scatter(ae, am, s=12, alpha=0.8)
    ax[1].set_xlabel(r'Measured asymmetry (%)')
    ax[1].set_ylabel(r'Model asymmetry (%)')
    _panel_label(ax[1], 'b')
    style_axes(ax[1])
    fig2.savefig(os.path.join(save_dir, 'sweep_scatter.png'), dpi=200, bbox_inches='tight')
    return dict(rmse=float(np.sqrt(np.mean(diff ** 2))), r2=1 - ss_res / ss_tot,
                meas_opt=(g1[ko], g2[ko]), model_opt=(g1[km], g2[km]),
                lost_gain=float(pe[ko] - pe[km]))


# --------------------------------------------------------------------------------------
# Turbulence-intensity comparison. The std file is the only direct check on the
# Crespo/Du accumulation, which is load-bearing -- switching TI accumulation off drops the
# wind-tunnel R2 from 0.95 to 0.67 -- and which has only ever been tested indirectly,
# through its effect on the velocity field.
#
# The model does not output TI. NuT_model computes it internally, so this rebuilds the SAME
# quantity from the solved farm: same correlation, same constants, same gates. It is
# therefore a test of the closure, not of a re-implementation of it.
# --------------------------------------------------------------------------------------

def model_ti_field(wind_farm, x_D, y_D, i_amb=I_AMB):
    """TI(x, y) at hub height exactly as NuT_model forms it.

    Iu^m = I_amb^m + sum_j [dI_j * U_j/U_0]^m  over rotors j with (x-x_j)/D_j >= 3 and
    |y-y_j| < 3 D_j, with dI_j = 0.73 a_j^0.8325 I_amb^-0.0325 ((x-x_j)/D_j)^-0.32, then
    TI = Iu/1.28. At each x every rotor upstream of x contributes, which is the
    generalisation of the per-station `up` list the solver builds segment by segment.
    """
    from simulation.utils import CRESPO_M
    m, IU_TO_TI = CRESPO_M, 1.28
    ts = wind_farm.turbines
    U0 = float(np.mean([t.Uhub for t in ts if abs(t.pos[0] - min(q.pos[0] for q in ts)) < 1e-9]))
    acc = np.full((len(y_D), len(x_D)), i_amb ** m)
    X, Y = np.meshgrid(x_D * D, y_D * D, indexing='xy')
    for t in ts:
        xi = (X - t.pos[0]) / t.D
        with np.errstate(invalid='ignore', divide='ignore'):
            dI = 0.73 * t.a ** 0.8325 * i_amb ** (-0.0325) * np.where(xi > 0, xi, np.nan) ** (-0.32)
        gate = (xi >= 3.0) & (np.abs(Y - t.pos[1]) < 3.0 * t.D)
        acc += np.where(gate, (np.nan_to_num(dI) * t.Uhub / U0) ** m, 0.0)
    # Returns I_u, NOT TI. I_amb accumulates in I_u space inside NuT_model
    # (Iu_sq = I_amb**m + ...) and is only divided by 1.28 at the last step to feed Du's
    # Eq (26), which wants a total turbulence intensity. The measured std file is I_u
    # (its far field reads 0.071, the paper's stated 7.1% hub-height value), so comparing
    # against it means NOT dividing. Doing so understated the model by a factor 1.28.
    del IU_TO_TI
    return acc ** (1.0 / m)


def validate_ti_field(case, yaw, save_dir='runs/validation_fullwake', n_grids=28,
                      x_stride=4, i_amb=I_AMB):
    exp = load_fullwake_case(case, 'std')
    x_D = exp['x_D'][::x_stride]
    y_D = exp['y_D']
    ti_exp = exp['z'][:, ::x_stride]
    sim = run_fullwake_case(yaw, n_grids=n_grids, x_D_max=float(np.nanmax(exp['x_D'])),
                            y_D_halfwidth=float(np.nanmax(np.abs(y_D))))
    ti_sim = model_ti_field(sim.wind_farm, x_D, y_D, i_amb=i_amb)

    msk = np.isfinite(ti_exp) & np.isfinite(ti_sim)
    d = ti_sim[msk] - ti_exp[msk]
    ss = np.sum((ti_exp[msk] - ti_exp[msk].mean()) ** 2)
    print('  case %s TI: RMSE=%.4f MAE=%.4f bias=%+.4f R2=%+.3f  exp mean %.4f model mean %.4f'
          % (case, np.sqrt(np.mean(d ** 2)), np.mean(np.abs(d)), d.mean(),
             1 - np.sum(d ** 2) / ss, ti_exp[msk].mean(), ti_sim[msk].mean()))

    plt.close('all')
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.4), sharex=True)
    style_figure(fig)
    lo, hi = 0.04, 0.22
    for ax, F, ttl in ((axes[0], ti_exp, 'Experiment $I_u$'),
                       (axes[1], ti_sim, 'Model $I_u$')):
        im = ax.pcolormesh(x_D, y_D, F, cmap='turbo', vmin=lo, vmax=hi, shading='nearest')
        fig.colorbar(im, ax=ax, label='$I_u$')
        ax.set_ylabel('$y/D$')
        style_axes(ax)
    im = axes[2].pcolormesh(x_D, y_D, ti_sim - ti_exp, cmap='RdBu_r', vmin=-0.10, vmax=0.10,
                            shading='nearest')
    fig.colorbar(im, ax=axes[2], label=r'$\Delta I_u$')
    axes[2].set_ylabel('$y/D$')
    axes[2].set_xlabel('$x/D$')
    style_axes(axes[2])
    for ax, L in zip(axes, 'abc'):
        _panel_label(ax, L)
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'ti_contour_%s.png' % case), dpi=200,
                bbox_inches='tight')

    fig2, axs = plt.subplots(1, 6, figsize=(8.7, 2.6), sharey=True, sharex=True)
    style_figure(fig2)
    for ax, xt in zip(axs, (2, 3, 7, 8, 12, 14)):
        i = int(np.argmin(np.abs(x_D - xt)))
        ax.plot(ti_exp[:, i], y_D, color='0.3', lw=1.1, label='Experiment')
        ax.plot(ti_sim[:, i], y_D, 'C0--', lw=1.1, label='Model')
        ax.set_title('$x/D = %d$' % xt, fontsize=7)
        ax.set_xlabel('$I_u$')
        style_axes(ax)
    axs[0].set_ylabel('$y/D$')
    axs[0].legend(fontsize=6, frameon=False)
    fig2.savefig(os.path.join(save_dir, 'ti_profiles_%s.png' % case), dpi=200,
                 bbox_inches='tight')
    return dict(rmse=float(np.sqrt(np.mean(d ** 2))), bias=float(d.mean()))


def ishihara_qian_ti_field(wind_farm, x_D, y_D, i_amb=I_AMB, y_centre=None):
    """Added-TI field of Ishihara & Qian (2018) JWEIA 177, 275-292, at hub height.

    Eq (35): dI_1 = G(C_T, I_a, x/D) * phi(r/sigma), with the spanwise function (37) a pair
    of Gaussians centred on the BLADE TIPS, r = +-D/2 -- "the maximum value of added
    turbulence intensity occurs at the tip of blade instead of the center of the rotor".
    That is the twin-peak structure the measurement shows and the current Crespo form cannot
    produce, since Crespo gives one number per rotor with no lateral shape at all.

    sigma is Ishihara & Qian's OWN width, Eq (28) -- per rotor, geometric, a function of
    C_T, I_a and (x - x_k) only. It needs no wake identification, so it behaves identically
    for oblique and irregular layouts; taking sigma from the solved deficit instead would
    reintroduce the per-wake attribution problem the age field already has.

    Combined across rotors with the same m = 2 quadrature the model already uses, which is
    also Ishihara & Qian's own rule: their Eq (34) adds the ambient and added standard
    deviations in quadrature. So only the SPATIAL DISTRIBUTION changes here, not the
    superposition.

    `y_centre`, if given, is the wake centreline in D at each x: the twin Gaussians are then
    centred on the DEFLECTED wake rather than on the rotor's own y_k. That matters only
    under yaw, and it reintroduces a dependence on the solved field (well defined for one
    column, ambiguous for overlapping wakes), so the self-contained rotor-centred form is
    the default. Ishihara & Qian supply their own analytic deflection, which is what a
    production implementation should use instead of reading the centreline off the march.

    Returns I_u, matching model_ti_field.
    """
    from simulation.utils import CRESPO_M
    m = CRESPO_M
    ts = wind_farm.turbines
    U0 = float(np.mean([t.Uhub for t in ts
                        if abs(t.pos[0] - min(q.pos[0] for q in ts)) < 1e-9]))
    acc = np.full((len(y_D), len(x_D)), i_amb ** m)
    X, Y = np.meshgrid(x_D * D, y_D * D, indexing='xy')
    YC = (np.tile(np.asarray(y_centre, float) * D, (len(y_D), 1))
          if y_centre is not None else None)
    for t in ts:
        ct, Dk = t.Ct, t.D
        xi = (X - t.pos[0]) / Dk
        # Eq (28): the same width as the velocity-deficit Gaussian
        kstar = 0.11 * ct ** 1.07 * i_amb ** 0.20
        eps = 0.23 * ct ** (-0.25) * i_amb ** 0.17
        sig = np.maximum(kstar * xi + eps, 1e-6)
        # Eqs (42)-(45): streamwise function, maximum added TI at the blade tip
        d = 2.3 * ct ** (-1.2)
        e = 1.0 * i_amb ** 0.1
        f = 0.7 * ct ** (-3.2) * i_amb ** (-0.45)
        G = 1.0 / (d + e * xi + f * (1.0 + np.maximum(xi, 0.0)) ** (-2))
        # Eqs (37)-(39): twin Gaussians at r = +-D/2, blended inside the rotor
        r = np.abs(Y - (t.pos[1] if YC is None else YC))
        rD = r / Dk
        k1 = np.where(rD <= 0.5, np.cos(np.pi / 2 * (rD - 0.5)) ** 2, 1.0)
        k2 = np.where(rD <= 0.5, np.cos(np.pi / 2 * (rD + 0.5)) ** 2, 0.0)
        phi = (k1 * np.exp(-((r - Dk / 2) ** 2) / (2 * (sig * Dk) ** 2))
               + k2 * np.exp(-((r + Dk / 2) ** 2) / (2 * (sig * Dk) ** 2)))
        acc += np.where(xi > 0, (G * phi * t.Uhub / U0) ** m, 0.0)
    return acc ** (1.0 / m)


def plot_ti_comparison(case, yaw, save_dir='runs/validation_fullwake', n_grids=28,
                       x_stride=4, i_amb=I_AMB):
    """Measured vs current (Crespo) vs Ishihara-Qian added TI, as contours and profiles.

    All three are I_u, so they are directly comparable -- the model's internal TI is
    I_u/1.28 and must NOT be used against this measurement.
    """
    ea = load_fullwake_case(case, 'avg')
    es = load_fullwake_case(case, 'std')
    x_D = es['x_D'][::x_stride]
    y_D = es['y_D']
    ti_exp = es['z'][:, ::x_stride]
    sim = run_fullwake_case(yaw, i_amb=i_amb, n_grids=n_grids,
                            x_D_max=float(np.nanmax(es['x_D'])),
                            y_D_halfwidth=float(np.nanmax(np.abs(y_D))))
    sl = extract_farm_hub_height_slice(sim.wind_farm, x_D * D, y_D * D)
    yc = wake_centre(y_D, sl['U'] / sl['Uh'])
    yc = np.where(np.isfinite(yc), yc, 0.0)

    cr = model_ti_field(sim.wind_farm, x_D, y_D, i_amb=i_amb)
    iq = ishihara_qian_ti_field(sim.wind_farm, x_D, y_D, i_amb=i_amb)
    iqd = ishihara_qian_ti_field(sim.wind_farm, x_D, y_D, i_amb=i_amb, y_centre=yc)
    sh = shear_ti_field(sim.wind_farm, x_D, y_D, sl['U'] / sl['Uh'], i_amb=i_amb)
    mc = marched_ti_field_cyl(sim.wind_farm, x_D, y_D, sl['U'] / sl['Uh'],
                              y_centre=yc, i_amb=i_amb)

    def stat(F):
        m = np.isfinite(ti_exp) & np.isfinite(F)
        d = F[m] - ti_exp[m]
        ss = np.sum((ti_exp[m] - ti_exp[m].mean()) ** 2)
        return np.sqrt(np.mean(d ** 2)), d.mean(), 1 - np.sum(d ** 2) / ss
    for lab, F in (('Crespo (current)', cr), ('Ishihara-Qian', iq),
                   ('Ishihara-Qian, deflected', iqd), ('shear, local equilib', sh),
                   ('marched Du budget (cyl)', mc)):
        r, b, r2 = stat(F)
        print('  case %s  %-26s RMSE=%.4f bias=%+.4f R2=%+.3f' % (case, lab, r, b, r2))

    os.makedirs(save_dir, exist_ok=True)
    plt.close('all')
    lo, hi = 0.04, 0.22
    fig, ax = plt.subplots(6, 1, figsize=(7.0, 10.0), sharex=True)
    style_figure(fig)
    for a, F, lbl in ((ax[0], ti_exp, 'Experiment'), (ax[1], cr, 'Crespo (current)'),
                      (ax[2], iqd, 'Ishihara-Qian'), (ax[3], mc, 'marched Du budget')):
        im = a.pcolormesh(x_D, y_D, F, cmap='turbo', vmin=lo, vmax=hi, shading='nearest')
        fig.colorbar(im, ax=a, label='$I_u$')
        a.set_ylabel('$y/D$')
        style_axes(a)
    for a, F in ((ax[4], iqd - ti_exp), (ax[5], mc - ti_exp)):
        im = a.pcolormesh(x_D, y_D, F, cmap='RdBu_r', vmin=-0.10, vmax=0.10,
                          shading='nearest')
        fig.colorbar(im, ax=a, label=r'$\Delta I_u$')
        a.set_ylabel('$y/D$')
        style_axes(a)
    ax[-1].set_xlabel('$x/D$')
    for a, L in zip(ax, 'abcdef'):
        _panel_label(a, L)
    fig.savefig(os.path.join(save_dir, 'ti_compare_contour_%s.png' % case), dpi=200,
                bbox_inches='tight')

    fig2, axs = plt.subplots(1, 6, figsize=(9.6, 2.8), sharey=True, sharex=True)
    style_figure(fig2)
    for a, xt in zip(axs, (2, 3, 7, 8, 12, 14)):
        i = int(np.argmin(np.abs(x_D - xt)))
        a.plot(ti_exp[:, i], y_D, color='0.25', lw=1.3, label='Experiment')
        a.plot(cr[:, i], y_D, 'C3--', lw=1.1, label='Crespo (current)')
        a.plot(iqd[:, i], y_D, 'C0-', lw=1.1, label='I-Q, deflected')
        a.plot(mc[:, i], y_D, 'C2-', lw=1.4, label='marched Du budget')
        a.set_title('$x/D = %d$' % xt, fontsize=7)
        a.set_xlabel('$I_u$')
        style_axes(a)
    axs[0].set_ylabel('$y/D$')
    axs[0].legend(fontsize=5.5, frameon=False)
    fig2.savefig(os.path.join(save_dir, 'ti_compare_profiles_%s.png' % case), dpi=200,
                 bbox_inches='tight')


def shear_ti_field(wind_farm, x_D, y_D, U_norm, i_amb=I_AMB):
    """Added TI from Du's wake-added TKE budget at LOCAL EQUILIBRIUM.

    Du et al. (2025) Eq (13) is
        U0 d<k_w>/dx - diffusion + (nu_t/Psi)<k_w> = nu_t * U_rho^2 ,
    with U_rho = dU/dr. Dropping advection and diffusion leaves production = dissipation,
    i.e. <k_w> = Psi(x) * (dU/dr)^2 -- nu_t cancels. Their Eq (28) gives
        Psi/D^2 = 0.67 * (0.2 TI + 0.015)^2 / (0.4 TI + 0.010) * x/D .
    Then TI_added = sqrt(2 k_w/3)/U0 and I_u = 1.28 TI, so

        I_u,added = 1.28 * sqrt(2/3) * sqrt(Psi) * |dU/dr| / U0 .

    The attraction is that the lateral SHAPE is not assumed: it comes from the shear of the
    solved deficit, so the twin peaks sit wherever the shear layers actually are, the two
    peaks can differ in height, and the width follows the real wake rather than a fitted
    sigma. A local gradient also needs no wake attribution, so unlike sigma or the wake
    centre it is unaffected by oblique or irregular layouts.

    Two known weaknesses, both structural rather than calibration:
      * at the wake centreline dU/dr -> 0, so this predicts ~zero added TI there, whereas
        the measurement shows a finite floor -- that turbulence is DIFFUSED in, which is
        exactly the term dropped above;
      * x is taken from the nearest upstream rotor, the same construct as the wake-age
        field, so it inherits that ambiguity where wakes overlap.

    CAVEAT on provenance: the coefficients inside Eq (28) were read off that equation
    rather than from a printed fit of k_lm and C_eps, so 0.2/0.015/0.4/0.010 carry more
    uncertainty than the 0.67.
    """
    from simulation.utils import CRESPO_M
    m, IU_TO_TI = CRESPO_M, 1.28
    ts = wind_farm.turbines
    U0 = float(np.mean([t.Uhub for t in ts
                        if abs(t.pos[0] - min(q.pos[0] for q in ts)) < 1e-9]))
    ti = i_amb / IU_TO_TI
    psi_over_D2 = 0.67 * (0.2 * ti + 0.015) ** 2 / (0.4 * ti + 0.010)

    X, Y = np.meshgrid(x_D * D, y_D * D, indexing='xy')
    # distance to the nearest UPSTREAM rotor, as the wake-age construct does
    age = np.full_like(X, np.inf)
    for t in ts:
        dx = X - t.pos[0]
        age = np.where((dx > 0) & (dx < age), dx, age)
    age = np.where(np.isfinite(age), age, 0.0)
    psi = psi_over_D2 * (age / D) * D ** 2

    dUdy = np.gradient(U_norm * U0, y_D * D, axis=0)          # physical dU/dy
    added = IU_TO_TI * np.sqrt(2.0 / 3.0) * np.sqrt(np.maximum(psi, 0)) * np.abs(dUdy) / U0
    return (i_amb ** m + added ** m) ** (1.0 / m)


def marched_ti_field(wind_farm, x_D, y_D, U_norm, i_amb=I_AMB, safety=0.4):
    """Added TI by MARCHING Du's wake-added TKE budget, Eq (13), keeping diffusion.

        U0 dk_w/dx = nu_t d2k_w/dy2 - (nu_t/Psi) k_w + nu_t (dU/dy)^2

    Du's Eq (14) is the analytic solution of this same equation, so marching it tests the
    same physics while avoiding that solution's modified-Bessel approximation, its
    unreconciled Psi vs psi, and the undefined delta in its integration limit.

    Two things this buys over the local-equilibrium form (shear_ti_field):
      * the diffusion term spreads production away from the shear layers, which is exactly
        what should cure the near-wake over-peak and fill the centreline;
      * k_w is TRANSPORTED, so several rotors simply add production to a field already
        being carried -- no Crespo p-norm superposition rule is needed at all, and no
        wake-age bookkeeping either, since x never appears except through nu_t and Psi.

    Approximations in THIS test, not in the idea: the diffusion is Cartesian in y rather
    than Du's cylindrical (1/r)d/dr(r dk/dr), which differs near the axis; nu_t and Psi use
    the ambient TI as Du intends (a case scalar), not an accumulated one.
    """
    from simulation.utils import CRESPO_M
    m, IU_TO_TI = CRESPO_M, 1.28
    kv1, kv0, xv1 = 0.05, 0.001, 0.5
    ts = wind_farm.turbines
    U0 = float(np.mean([t.Uhub for t in ts
                        if abs(t.pos[0] - min(q.pos[0] for q in ts)) < 1e-9]))
    ti = i_amb / IU_TO_TI
    x0 = min(t.pos[0] for t in ts)
    psi_D2 = 0.67 * (0.2 * ti + 0.015) ** 2 / (0.4 * ti + 0.010)

    y_m = y_D * D
    dy = float(np.mean(np.diff(y_m)))
    U = U_norm * U0
    dUdy = np.gradient(U, y_m, axis=0)

    kw = np.zeros((len(y_D), len(x_D)))
    k = np.zeros(len(y_D))
    for i in range(1, len(x_D)):
        xa, xb = (x_D[i - 1] - x0 / D), (x_D[i] - x0 / D)
        dx_tot = (x_D[i] - x_D[i - 1]) * D
        # Explicit diffusion needs nu_t*dx/(U0*dy^2) <= 1/2; substep to hold it at `safety`.
        # At n_grids=28 one PIV station already sits at 0.67, which diverges -- the first
        # attempt at this test blew up to RMSE ~5e14 for exactly that reason, while still
        # giving the right answer at x/D=2 before it did.
        nut_max = max(kv1 * ti - kv0, 0.0) * min(max(xb, 1e-6), xv1 / ti) * U0 * D
        nsub = int(max(1, np.ceil(nut_max * dx_tot / (U0 * dy ** 2 * safety))))
        for sb in range(nsub):
            xi = max(xa + (xb - xa) * (sb + 0.5) / nsub, 1e-6)
            nut = max(kv1 * ti - kv0, 0.0) * min(xi, xv1 / ti) * U0 * D
            psi = max(psi_D2 * xi * D ** 2, 1e-12)
            prod = nut * (0.5 * (dUdy[:, i - 1] + dUdy[:, i])) ** 2
            lap = np.zeros_like(k)
            lap[1:-1] = (k[2:] - 2 * k[1:-1] + k[:-2]) / dy ** 2
            k = k + (dx_tot / nsub / U0) * (nut * lap - (nut / psi) * k + prod)
            k = np.maximum(k, 0.0)
        kw[:, i] = k
    added = IU_TO_TI * np.sqrt(2.0 / 3.0 * kw) / U0
    return (i_amb ** m + added ** m) ** (1.0 / m)


def marched_ti_field_cyl(wind_farm, x_D, y_D, U_norm, y_centre=None, i_amb=I_AMB,
                         safety=0.4, age_from='rotor'):
    """Du Eq (13) marched in the CORRECT cylindrical frame.

    The diffusion term is Du's own, (nu_t/r) d/dr(r dk_w/dr) = d2k/dr2 + (1/r) dk/dr, not
    the Cartesian d2k/dy2 the first test used. Solved on a radial grid about the wake axis
    with dk/dr = 0 at r = 0 and k = 0 at r_max, then mapped back to y through
    r = |y - y_c|. At the axis the operator's limit is 2 d2k/dr2, handled separately.

    This is faithful to Du's formulation, which is axisymmetric in <k_w>_theta plus a
    ground correction. Two consequences worth being explicit about:
      * it needs the wake AXIS, so it inherits a dependence on the solved deflection --
        intrinsic to an axisymmetric model, not an artefact of this implementation;
      * being axisymmetric it CANNOT reproduce unequal lateral peaks, which the
        measurement shows. Du patches the vertical asymmetry with delta(x, r, theta); at
        hub height that term is small and symmetric in +-y, so it cannot help here either.

    Production is the azimuthal proxy available from a hub-height cut: (dU/dr)^2 averaged
    over the two sides y_c +- r.
    """
    from simulation.utils import CRESPO_M
    m, IU_TO_TI = CRESPO_M, 1.28
    kv1, kv0, xv1 = 0.05, 0.001, 0.5
    ts = wind_farm.turbines
    U0 = float(np.mean([t.Uhub for t in ts
                        if abs(t.pos[0] - min(q.pos[0] for q in ts)) < 1e-9]))
    ti = i_amb / IU_TO_TI
    psi_D2 = 0.67 * (0.2 * ti + 0.015) ** 2 / (0.4 * ti + 0.010)

    # WAKE AGE, not distance from the farm inlet. Psi = c*l_m^2/C_eps grows with the
    # mixing length, i.e. with the age of the shear layer -- and a shear layer restarts at
    # every rotor. Referencing Psi to the inlet instead made the amplitude (~sqrt(Psi))
    # grow without bound down the row: measured on case C, the peak I_u error went -1% in
    # bay 1 to +20% in bay 2 and +28% in bay 3, while Psi/D^2 ran 0.028 -> 0.099 -> 0.170.
    # This is the same per-wake-origin problem the nu_T age field has; a real
    # implementation should use the carried age, this test uses the geometric nearest
    # upstream rotor.
    xs = np.array(sorted(t.pos[0] for t in ts))

    def _age(x_abs):
        up = xs[xs < x_abs - 1e-9]
        return (x_abs - up[-1]) if len(up) else max(x_abs - xs[0], 1e-6)

    y_m = y_D * D
    dr = abs(float(np.mean(np.diff(y_m))))
    r = np.arange(0.0, float(np.abs(y_m).max()) + dr, dr)
    U = U_norm * U0
    dUdy = np.gradient(U, y_m, axis=0)
    yc = (np.zeros(len(x_D)) if y_centre is None
          else np.asarray(y_centre, float) * D)

    kw = np.zeros((len(y_D), len(x_D)))
    k = np.zeros(len(r))
    for i in range(1, len(x_D)):
        if age_from == 'inlet':
            xa, xb = (x_D[i - 1] - xs[0] / D), (x_D[i] - xs[0] / D)
        else:
            xa = _age(x_D[i - 1] * D) / D
            xb = _age(x_D[i] * D) / D
        dx_tot = (x_D[i] - x_D[i - 1]) * D
        g = 0.5 * (dUdy[:, i - 1] + dUdy[:, i])
        # azimuthal proxy: (dU/dr)^2 averaged over the two sides of the wake axis
        gp = np.interp(yc[i] + r, y_m, g)
        gm = np.interp(yc[i] - r, y_m, g)
        prod_shape = 0.5 * (gp ** 2 + gm ** 2)
        nut_max = max(kv1 * ti - kv0, 0.0) * min(max(xb, 1e-6), xv1 / ti) * U0 * D
        nsub = int(max(1, np.ceil(nut_max * dx_tot / (U0 * dr ** 2 * safety))))
        for sb in range(nsub):
            xi = max(xa + (xb - xa) * (sb + 0.5) / nsub, 1e-6)
            nut = max(kv1 * ti - kv0, 0.0) * min(xi, xv1 / ti) * U0 * D
            psi = max(psi_D2 * xi * D ** 2, 1e-12)
            lap = np.zeros_like(k)
            lap[0] = 2.0 * 2.0 * (k[1] - k[0]) / dr ** 2          # axis limit, dk/dr = 0
            lap[1:-1] = ((k[2:] - 2 * k[1:-1] + k[:-2]) / dr ** 2
                         + (k[2:] - k[:-2]) / (2 * dr * r[1:-1]))
            k = k + (dx_tot / nsub / U0) * (nut * lap - (nut / psi) * k
                                            + nut * prod_shape)
            k = np.maximum(k, 0.0)
            k[-1] = 0.0
        kw[:, i] = np.interp(np.abs(y_m - yc[i]), r, k)
    added = IU_TO_TI * np.sqrt(2.0 / 3.0 * kw) / U0
    return (i_amb ** m + added ** m) ** (1.0 / m)


def hub_height_age(wind_farm, x_D, y_D, n_grids=28, hard=False):
    """The carried wake-age field A(x, y) at hub height, in metres.

    Closed form of dA/dx = 1 with a reset inside each shedding rotor's disc:
        A <- (1 - chi_k) * A   at rotor k,   A += dx   between rotors.
    chi_k is the GAUSSIAN-SMOOTHED disc the solver actually uses (sigma = 1.5 cells), so
    the reset is PARTIAL at the disc edge: the wake core restarts while the fringes keep
    ageing. That is the distinction a hard nearest-upstream-rotor reset gets wrong -- it
    resets the whole plane, which is why Psi then under-accumulates down the row.

    `hard=True` gives the hard-disc version for comparison.
    """
    from scipy.ndimage import gaussian_filter1d
    y_m = y_D * D
    A = np.zeros((len(y_D), len(x_D)))
    a = np.zeros(len(y_D))
    xs = sorted(wind_farm.turbines, key=lambda t: t.pos[0])
    sigma_cells = 1.5 * (D / n_grids) / abs(float(np.mean(np.diff(y_m))))
    nxt = 0
    for i in range(len(x_D)):
        if i > 0:
            a = a + (x_D[i] - x_D[i - 1]) * D
        while nxt < len(xs) and xs[nxt].pos[0] <= x_D[i] * D + 1e-9:
            t = xs[nxt]
            inside = (np.abs(y_m - t.pos[1] + t.Yoffset) / max(np.cos(t.beta), 1e-6)
                      <= t.D / 2.0).astype(float)
            chi = inside if hard else gaussian_filter1d(inside, sigma_cells, mode='nearest')
            a = (1.0 - np.clip(chi, 0.0, 1.0)) * a
            nxt += 1
        A[:, i] = a
    return A


def marched_ti_field_age(wind_farm, x_D, y_D, U_norm, age, y_centre=None, i_amb=I_AMB,
                         safety=0.4):
    """Du Eq (13) marched with a PER-CELL wake age setting nu_t and Psi.

    Same equation and same cylindrical operator as marched_ti_field_cyl; the only change is
    that nu_t and Psi are evaluated on the carried age A(y) rather than on one scalar
    distance per station.

    Test approximation: the age enters production and dissipation per cell, but the
    diffusion COEFFICIENT uses the wake-core age, because a radially varying nu_t needs the
    conservative form div(nu_t grad k) rather than nu_t*lap(k). Flagged rather than hidden.
    """
    from simulation.utils import CRESPO_M
    m, IU_TO_TI = CRESPO_M, 1.28
    kv1, kv0, xv1 = 0.05, 0.001, 0.5
    ts = wind_farm.turbines
    U0 = float(np.mean([t.Uhub for t in ts
                        if abs(t.pos[0] - min(q.pos[0] for q in ts)) < 1e-9]))
    ti = i_amb / IU_TO_TI
    psi_D2 = 0.67 * (0.2 * ti + 0.015) ** 2 / (0.4 * ti + 0.010)
    nut_of = lambda xi: max(kv1 * ti - kv0, 0.0) * min(max(xi, 1e-6), xv1 / ti) * U0 * D

    y_m = y_D * D
    dr = abs(float(np.mean(np.diff(y_m))))
    r = np.arange(0.0, float(np.abs(y_m).max()) + dr, dr)
    U = U_norm * U0
    dUdy = np.gradient(U, y_m, axis=0)
    yc = np.zeros(len(x_D)) if y_centre is None else np.asarray(y_centre, float) * D

    kw = np.zeros((len(y_D), len(x_D)))
    k = np.zeros(len(r))
    for i in range(1, len(x_D)):
        dx_tot = (x_D[i] - x_D[i - 1]) * D
        g = 0.5 * (dUdy[:, i - 1] + dUdy[:, i])
        gp, gm = np.interp(yc[i] + r, y_m, g), np.interp(yc[i] - r, y_m, g)
        prod_shape = 0.5 * (gp ** 2 + gm ** 2)
        ag = 0.5 * (np.interp(yc[i] + r, y_m, age[:, i]) +
                    np.interp(yc[i] - r, y_m, age[:, i])) / D      # age in D, per radius
        nut_r = np.array([nut_of(v) for v in ag])
        psi_r = np.maximum(psi_D2 * np.maximum(ag, 1e-6) * D ** 2, 1e-12)
        core = float(np.interp(0.0, r, ag))
        nut_core = nut_of(core)
        nsub = int(max(1, np.ceil(nut_core * dx_tot / (U0 * dr ** 2 * safety))))
        for sb in range(nsub):
            lap = np.zeros_like(k)
            lap[0] = 4.0 * (k[1] - k[0]) / dr ** 2
            lap[1:-1] = ((k[2:] - 2 * k[1:-1] + k[:-2]) / dr ** 2
                         + (k[2:] - k[:-2]) / (2 * dr * r[1:-1]))
            k = k + (dx_tot / nsub / U0) * (nut_core * lap - (nut_r / psi_r) * k
                                            + nut_r * prod_shape)
            k = np.maximum(k, 0.0)
            k[-1] = 0.0
        kw[:, i] = np.interp(np.abs(y_m - yc[i]), r, k)
    added = IU_TO_TI * np.sqrt(2.0 / 3.0 * kw) / U0
    return (i_amb ** m + added ** m) ** (1.0 / m)
