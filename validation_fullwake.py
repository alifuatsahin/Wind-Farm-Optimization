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

#: Hold log(Zh/z0) at the model's full-scale default, exactly as _base_case does, so the
#: inflow profile SHAPE is unchanged and z0 does not become a second free parameter.
#:
#: THIS USED TO SAY "z0 is not measured for this tunnel". That is FALSE: Zong &
#: Porte-Agel (2020) Sec 2 fit the same facility's boundary layer and report u* = 0.22
#: m/s, a viscous length of 0.068 mm and z0 = 0.7 mm. The caveat that IS true is narrower
#: -- their boundary layer runs at I_amb = 0.055 and this dataset at 0.070, so 0.7 mm is
#: better sourced than the back-scaled 4.17e-5 without being exact for this inflow.
#:
#: AND IT MATTERS MORE THAN AN INFLOW CONSTANT SHOULD, because z0 does DOUBLE DUTY: it
#: sets the log profile AND, through turbine_physics.abl_nu_slope (k = 0.4/ln(Zh/z0),
#: nu ~ k^2), the vortex core diffusivity that decides how long the CVP survives and
#: therefore how far the wake steers. Those are different measurements. Between the two
#: values the lever on nu_c is 2.38x, and measured on this row at yaw (25, 15, 0) the
#: STEERING GAIN CHANGES SIGN -- +1.63% at 4.17e-5 against -0.93% at 7.0e-4, off a
#: deflection that moves only 4.7%. Do not read a yaw benefit from this model without
#: quoting z0 alongside it; see memory z0-double-duty-flips-the-gain.
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
    """Per-rotor power in the model's own terms: P ~ Cp(beta) * Uhub3^3.

    params.Cp already carries the cos^CP_YAW_EXP yaw factor, so this is the same quantity the
    objective maximises -- not the hub-height line proxy compute_power_metrics builds for
    datasets that do not measure power. Only ratios are compared, so the omitted
    0.5*rho*A cancels.

    Uhub3 is the CUBIC rotor-equivalent speed, Zong Eq (4.11)'s disc integral of U^3,
    NOT the arithmetic disc mean -- see turbine_physics.rotor_equivalent_cubic for why
    the distinction matters most at the partial wake overlap a steering optimum sits in.
    """
    return np.array([t.Cp * t.Uhub3 ** 3 for t in wind_farm.turbines])


def run_fullwake_case(yaw, i_amb=I_AMB, n_grids=20, x_D_max=15.0, y_D_halfwidth=2.1,
                      **field_kw):
    """Three-turbine row at the measured spacing.

    Runs the SHAPIRO ABL vortex viscosity, which is FieldConfig's default. This docstring
    used to claim the opposite -- "deliberately does NOT pass vortex_nu='abl' ... so the
    default (Squire) applies" -- but `vortex_nu` defaults to 'abl' (config.py) and
    turbine_physics.vortex_adapter switches to Squire only on an explicit 'squire', so the
    ABL form is what has always run here, on both wind-tunnel rigs. The CODE was right and
    the comment was wrong: 'abl' is the adopted closure (it wins on deflection and field;
    see memory shapiro-abl-vortex-viscosity), and its u*/z0 scaling is as valid in a
    tunnel boundary layer as in an atmospheric one -- Zong & Porte-Agel (2020) Sec 2 fit
    u* = 0.22 m/s and z0 = 0.7 mm for this very rig. Pass vortex_nu='squire' for the
    ablation.

    `field_kw` goes straight to FieldConfig, for sweeping things like merge_threshold."""
    wf = WindFarmConfig(
        pos=np.array([[xd * D, 0.0, 0.0] for xd in POSITIONS_D], dtype=float),
        D=np.array([D]), Zhub=np.array([ZHUB]), Ct=np.array([CT]), Cp=np.array([CP]),
        yaw=np.array(yaw, dtype=float), TSR=np.array([TSR]),
        nacelle_D=np.array([10.0 / 150.0]),  # WIRE-01 Maxon DCX10L, Zong Sec 2
        dist_type='m',
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
# Turbulence-intensity comparison. The std file is the only DIRECT check on the marched
# wake-added TKE, which is load-bearing twice over: it sets the eddy viscosity of every
# waked rotor (Du's Eq (26) reads the TI incident on each station off this same k_w), and
# until this file it had only ever been tested indirectly, through the velocity field.
#
# solver_ti_field reads TI back from the solver's own k_w, so this compares the quantity
# the closure actually used -- not a post-hoc reconstruction of it.
# --------------------------------------------------------------------------------------

def validate_ti_field(case, yaw, save_dir='runs/validation_fullwake', n_grids=28,
                      x_stride=4, i_amb=I_AMB):
    exp = load_fullwake_case(case, 'std')
    x_D = exp['x_D'][::x_stride]
    y_D = exp['y_D']
    ti_exp = exp['z'][:, ::x_stride]
    # i_amb MUST reach the simulation, not only the TI read-back. It did not: the
    # field was marched at the module default while solver_ti_field interpreted it
    # at the passed value, so any call with a non-default i_amb silently mismatched
    # the two. Latent at I_AMB = 0.070, wrong for anything else.
    sim = run_fullwake_case(yaw, i_amb=i_amb, n_grids=n_grids,
                            x_D_max=float(np.nanmax(exp['x_D'])),
                            y_D_halfwidth=float(np.nanmax(np.abs(y_D))))
    ti_sim = solver_ti_field(sim.wind_farm, x_D, y_D, i_amb=i_amb)

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
                       (axes[1], ti_sim, 'Model $I_u$ (marched $k_w$)')):
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


#: I_u/TI for WAKE-ADDED turbulence. NOT the ambient 1.28 (utils.IU_TO_TI): that ratio is
#: a property of the atmospheric surface layer, and wake-added turbulence is very nearly
#: isotropic. MEASURED from the 1240 hub-height snapshots in data/ABL_flow_3turbine.h5, by
#: taking sigma_u and sigma_v over the time axis and subtracting the ambient in quadrature:
#:
#:     x/D        9      12      15      18      21
#:     sigma_v+/sigma_u+   0.823   0.880   1.052   0.945   0.999
#:     I_u/TI            1.129   1.085   0.966   1.038   1.001
#:
#: using I_u/TI = 1/sqrt((1 + 2 r^2)/3) with sigma_w+ taken equal to sigma_v+, the wake
#: shear layer being roughly axisymmetric. The mean is 1.04; 1.0 is used, the scatter
#: being larger than the difference. The same calculation on the AMBIENT, with the
#: Panofsky-Dutton sigma_w/sigma_u = 0.5 that van der Laan et al. (2015) Eq (8) adopts,
#: returns 1.31 against the coded 1.28 -- which is what validates the method here.
#:
#: Only sigma_u and sigma_v are in that dataset, so sigma_w+ = sigma_v+ is an assumption,
#: not a measurement. Re-derive it if a three-component wake dataset turns up.
IU_TO_TI_WAKE = 1.0


def solver_ti_field(wind_farm, x_D, y_D, i_amb=I_AMB):
    """The TI the SOLVER actually used, read back from the marched k_w field.

    Everything else in this module reconstructs a TI post hoc from rotor positions. Now
    that k_w is carried in the march, the model has its own internal TI and that is what
    should be plotted against the measurement -- a reconstruction can agree with the data
    while the solver uses something else.

    The ambient and the wake-added parts are converted to I_u with DIFFERENT anisotropy
    ratios, 1.28 and IU_TO_TI_WAKE: see that constant. Using the ambient 1.28 for both
    inflated the model's wake-added I_u by 28%, which flattered the shipped closure --
    on these three cases it moved its TI bias from -0.008/-0.006/-0.007 (8% LOW) to
    +0.000/+0.001/+0.000, i.e. the error was cancelled by the scaling rather than absent.
    """
    from simulation.field_ops import _source_turbine
    ts = wind_farm.turbines
    U0 = float(np.mean([t.Uhub for t in ts
                        if abs(t.pos[0] - min(q.pos[0] for q in ts)) < 1e-9]))
    out = np.full((len(y_D), len(x_D)), i_amb)
    for i, xd in enumerate(x_D):
        x_m = xd * D
        src = _source_turbine(ts, x_m)
        if src is None or src.wake_field is None:
            continue
        frames = src.wake_field
        j = int(np.argmin([abs(float(f.X) - (x_m - src.pos[0])) for f in frames]))
        kw = np.asarray(frames[j].kw)
        if kw.size == 0:
            continue
        zc = np.asarray(src.zloc)[0, :]
        kz = int(np.argmin(np.abs(zc - src.Zhub)))          # hub-height row
        y_abs = np.asarray(src.yloc)[:, kz] + src.pos[1]
        kw_line = np.interp(y_D * D, y_abs, kw[:, kz])
        iu_add = IU_TO_TI_WAKE * np.sqrt(np.maximum(2.0 / 3.0 * kw_line, 0.0)) / U0
        out[:, i] = np.sqrt(i_amb ** 2 + iu_add ** 2)
    return out
