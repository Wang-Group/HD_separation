"""Render the current three non-eta10 categories without changing adopted records."""
import hashlib
import json
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from extract_eta10 import HERE, RAW, read_csv, load_curve

OUT = HERE / 'estimate_categories_review'
CATEGORIES = {
    'nearest_point_estimate': ('Nearest-point estimates', 'nearest_point_estimates', '#cc5e19', 'D'),
    'rising_point': ('Rising-point estimates', 'rising_point_estimates', '#087e78', 'D'),
    'above_target_no_crossing': ('No accepted result', 'no_accepted_result', '#555555', 'x'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_categories():
    manifest = {r['curve_id']: r for r in read_csv(RAW / 'manifest.csv')}
    groups = {status: [] for status in CATEGORIES}
    for row in read_csv(HERE / 'eta10_curves.csv'):
        status = row['status']
        if status not in groups:
            continue
        record = manifest[row['curve_id']]
        assert record['sha256'] == row['sha256']
        points, eta, j, area, factor, basis = load_curve(record)
        eta, j = np.array(eta), np.array(j)
        assert row['eta10_mV'] == row['scan_limit_mV'] == ''
        if status == 'above_target_no_crossing':
            k = min(range(len(j)), key=lambda k: (abs(j[k] + 10), -eta[k]))
            potential = float(row['nearest_candidate_potential_mV'])
            current = float(row['nearest_candidate_j_mA_cm2'])
            assert np.all(j < -10) and j[k] <= -20
        else:
            k = next(k for k, p in enumerate(points) if p[0] == int(row['raw_line_before']))
            potential = float(row['estimate_potential_mV'])
            current = float(row['estimate_j_mA_cm2'])
            assert -20 < j[k] < -10
        assert abs(eta[k] - potential) < 1e-6 and abs(j[k] - current) < 1e-6
        if status == 'nearest_point_estimate':
            assert np.all(j < -10)
            assert abs(abs(j[k] + 10) - min(abs(j + 10))) < 1e-9
        groups[status].append(dict(row, eta=eta, j=j, point_eta=potential, point_j=current,
            area=area, basis=basis, raw_line=points[k][0]))
    summary = json.loads((HERE / 'extraction_summary.json').read_text(encoding='utf-8'))
    assert all(len(rows) == summary['counts'][status] for status, rows in groups.items())
    return groups


def draw(ax, curve, low=False, compact=False):
    eta, j = curve['eta'], curve['j']
    _, _, color, marker = CATEGORIES[curve['status']]
    ax.plot(eta, j, color='#176b9a', lw=1.0 if compact else 1.5)
    ax.axhline(-10, color='#444444', ls='--', lw=.85)
    ax.axhline(-20, color='#999999', ls=':', lw=.85)
    ax.axvline(0, color='#cccccc', lw=.7)
    ax.plot(curve['point_eta'], curve['point_j'], marker=marker, ms=5 if compact else 9,
            mfc='white', mec=color, mew=1.5, color=color, linestyle='none', zorder=5)
    if curve['status'] == 'rising_point' and curve['largest_crossing_mV']:
        ax.plot(float(curve['largest_crossing_mV']), -10, marker='o', ms=4 if compact else 6,
                mfc='white', mec='#777777', mew=1, linestyle='none', zorder=4)
    left, right = float(eta.min()), float(eta.max())
    if low:
        bottom = min(-25, 5 * math.floor((curve['point_j'] - max(4, abs(curve['point_j']) * .2)) / 5))
        visible = eta[j >= bottom]
        assert len(visible)
        right = min(right, max(float(visible.max()), curve['point_eta']) + 25)
        ax.set_ylim(bottom, 3)
    else:
        bottom = min(-25, float(j.min()) - max(2, np.ptp(j) * .05))
        ax.set_ylim(bottom, max(3, float(j.max()) + 2))
    margin = max(5, (right - left) * .035)
    ax.set_xlim(left - margin, right + margin)
    assert ax.get_xlim()[0] < curve['point_eta'] < ax.get_xlim()[1]
    assert ax.get_ylim()[0] < curve['point_j'] < ax.get_ylim()[1]
    ax.grid(alpha=.23, lw=.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(labelsize=8 if compact else 12)
    if not compact:
        ax.set_xlabel('Overpotential (mV)', fontsize=13)
        ax.set_ylabel(r'Geometric current density (mA cm$^{-2}$)', fontsize=13)
        title = 'Low-current view' if low else 'Full recorded range'
        if low and bottom < -25:
            title += ' (expanded to include diagnostic point)'
        ax.set_title(title, fontsize=12.5, pad=12)


def page(curve, number, total):
    title, _, color, marker = CATEGORIES[curve['status']]
    diagnostic = curve['status'] == 'above_target_no_crossing'
    label = 'Closest point (diagnostic only)' if diagnostic else title[:-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.subplots_adjust(left=.075, right=.98, bottom=.24, top=.75, wspace=.27)
    for ax, low in zip(axes, (False, True)):
        draw(ax, curve, low)
    fig.suptitle(f'{number:02d}/{total}  {curve["sample"]} | {Path(curve["file"]).name}',
                 fontsize=16, fontweight='bold', y=.975)
    fig.text(.5, .913, f'{label}: {curve["point_eta"]:.2f} mV at '
             + rf'{curve["point_j"]:.2f} mA cm$^{{-2}}$', ha='center', fontsize=13)
    note = ('No target crossing; closest recorded |j| is at least 20 mA cm$^{-2}$. No adopted value.'
            if diagnostic else r'Adopted estimate, not measured $\eta_{10}$. Potential and current density are retained separately.')
    fig.text(.5, .86, note, ha='center', fontsize=10.5, color='#555555')
    handles = [Line2D([], [], color='#176b9a', lw=1.5, label='Recorded LSV'),
        Line2D([], [], color=color, marker=marker, mfc='white', mew=1.5, ms=7,
               linestyle='none', label='Diagnostic closest point' if diagnostic else 'Adopted estimate'),
        Line2D([], [], color='#444444', ls='--', label=r'$j=-10$ mA cm$^{-2}$'),
        Line2D([], [], color='#999999', ls=':', label=r'$j=-20$ mA cm$^{-2}$')]
    if curve['status'] == 'rising_point' and curve['largest_crossing_mV']:
        handles.append(Line2D([], [], color='#777777', marker='o', mfc='white', linestyle='none',
                              label='Crossing (not adopted)'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .12),
               ncol=len(handles), frameon=False, fontsize=10)
    source = textwrap.fill('Source: ' + curve['source_relative_path'], width=150)
    fig.text(.075, .09, source, fontsize=9, va='top')
    if curve['basis'] == 'assumed_1cm2_unverified':
        area_note = r'Area: 1 cm$^2$ (assumed, not independently confirmed)'
    elif curve['area'] is not None:
        area_note = rf'Area: {curve["area"]:g} cm$^2$ (manifest override)'
    else:
        area_note = 'Exported current density; no further area division'
    fig.text(.075, .029, f'Raw TXT line: {curve["raw_line"]}    ' + area_note, fontsize=9, color='#555555')
    return fig


def overview(status, curves):
    title, stem, _, _ = CATEGORIES[status]
    nrows = math.ceil(len(curves) / 4)
    height = nrows * 2.65 + 1.45
    fig, axes = plt.subplots(nrows, 4, figsize=(17, height), squeeze=False)
    fig.subplots_adjust(left=.055, right=.985, top=1 - 1.2 / height,
                        bottom=.65 / height, wspace=.38, hspace=.62)
    for number, (ax, curve) in enumerate(zip(axes.flat, curves), 1):
        draw(ax, curve, low=True, compact=True)
        ax.set_title(f'{number:02d} {curve["sample"]}\n{curve["point_eta"]:.2f} mV; '
                     + rf'{curve["point_j"]:.2f} mA cm$^{{-2}}$', fontsize=9.5, pad=5)
    for ax in list(axes.flat)[len(curves):]:
        ax.set_visible(False)
    fig.suptitle(f'{title} | {len(curves)} curves', fontsize=19, y=1 - .14 / height)
    note = ('Marked points are diagnostic only; no accepted value' if status == 'above_target_no_crossing'
            else 'Marked points are adopted estimates, not measured eta10')
    fig.text(.5, 1 - .52 / height, note, ha='center', va='top', fontsize=11)
    fig.supxlabel('Overpotential (mV); panel ranges vary', y=.1 / height, fontsize=12)
    fig.supylabel(r'Geometric current density (mA cm$^{-2}$)', x=.009, fontsize=12)
    fig.savefig(OUT / f'{stem}_overview.png', dpi=170)
    plt.close(fig)


def main():
    protected = [HERE / name for name in ('eta10_curves.csv', 'eta10_samples.csv', 'extraction_summary.json',
        'lsv_rules.py', 'lsv_noise_exceptions.json')]
    protected.append(RAW / 'manifest.csv')
    before = {str(p): digest(p) for p in protected}
    font_manager.findfont('Arial', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'pdf.fonttype': 42,
        'axes.unicode_minus': True, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})
    groups = load_categories()
    OUT.mkdir(exist_ok=True)
    for status, curves in groups.items():
        title, stem, _, _ = CATEGORIES[status]
        with PdfPages(OUT / f'{stem}.pdf', metadata={'Title': title, 'CreationDate': None}) as pdf:
            for number, curve in enumerate(curves, 1):
                fig = page(curve, number, len(curves))
                pdf.savefig(fig)
                plt.close(fig)
        overview(status, curves)
        print(f'{title}: {len(curves)} pages and overview PNG', flush=True)
    assert before == {str(p): digest(p) for p in protected}
    verification = dict(counts={k: len(v) for k, v in groups.items()}, raw_hashes_verified=True,
        marked_points_verified=True, all_marked_points_within_axes=True, source_records_unchanged=True,
        font='Arial', ML_rerun=False, Figure4_changed=False, input_sha256=before,
        script_sha256=digest(Path(__file__)))
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    (OUT / 'README.md').write_text(
        '# Estimate category plots\n\n'
        f'Current snapshot: {len(groups["nearest_point_estimate"])} nearest-point estimates, '
        f'{len(groups["rising_point"])} rising-point estimates, and '
        f'{len(groups["above_target_no_crossing"])} no-accepted-result curves.\n'
        'Each category has a full-range/low-current PDF (one curve per page) and a low-current overview PNG.\n'
        'All plots use Arial and retain the extraction area normalization and potential convention.\n\n'
        'Nearest-point estimates are recorded points globally closest to -10 mA/cm2 when neither a crossing nor eligible rise exists.\n'
        'Rising-point estimates are automatically detected raw points associated with a later cathodic rise; they need not be closest to -10.\n'
        'When multiple rising candidates qualify, select the one with the largest overpotential.\n'
        'Both require -20 < selected j < -10 mA/cm2; neither is measured eta10 or proof of HER onset.\n'
        'The explicit Co-Ce repeat4 noise exception retains its crossing and is not included in these estimate PDFs.\n'
        'Gray circles on rising-point curves show the largest-overpotential crossing for comparison, not an adopted eta10.\n'
        'For no-accepted-result curves, gray crosses are diagnostic closest points only. Their |j| is at least 20 mA/cm2.\n'
        'Overview axis ranges vary; the low-current window expands when necessary to show the diagnostic point.\n\n'
        'Run `python plot_estimate_categories.py` from revision_figure. Only this output directory is written.\n'
        'No classifications, raw TXT files, ML outputs, or Figure 4/BO records are changed.\n', encoding='utf-8')
    print(OUT, flush=True)


if __name__ == '__main__':
    main()
