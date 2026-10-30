"""Plot the 35 unapproved nearest-point candidates without changing classifications."""
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from extract_eta10 import HERE, RAW, load_curve, read_csv, write_csv

REVIEW = HERE / 'rule_sync_20261025'
OUT = REVIEW / 'nearest_point_review'
SOURCE = REVIEW / 'unapproved_nearest_candidates.csv'
BLUE = '#176b9a'
ORANGE = '#d35d22'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidates():
    manifest = {r['curve_id']: r for r in read_csv(RAW / 'manifest.csv')}
    snapshot = OUT / 'before_approval/eta10_curves.csv'
    classified = {r['curve_id']: r for r in read_csv(snapshot if snapshot.exists() else HERE / 'eta10_curves.csv')}
    candidates = read_csv(SOURCE)
    assert len(candidates) == len({r['curve_id'] for r in candidates}) == 35
    curves = []
    for number, candidate in enumerate(candidates, 1):
        record = manifest[candidate['curve_id']]
        assert candidate['sha256'] == record['sha256']
        assert classified[candidate['curve_id']]['status'] == 'above_target_no_crossing'
        points, eta, j, area, factor, basis = load_curve(record)
        eta, j = np.asarray(eta), np.asarray(j)
        expected_eta = float(candidate['nearest_potential_mV'])
        expected_j = float(candidate['nearest_j_mA_cm2'])
        matches = np.flatnonzero((abs(eta - expected_eta) < 1e-6) & (abs(j - expected_j) < 1e-6))
        assert len(matches) >= 1, candidate['file']
        index = int(matches[0])
        assert -20 < j[index] < -10 and np.all(j < -10)
        assert abs(abs(j[index] + 10) - np.min(abs(j + 10))) < 1e-9
        curves.append(dict(number=number, sample=candidate['sample'], file=record['file'],
            source_relative_path=record['source_relative_path'], curve_id=record['curve_id'],
            sha256=record['sha256'], eta=eta, j=j, index=index, raw_line=points[index][0],
            matching_raw_lines=';'.join(str(points[int(k)][0]) for k in matches),
            candidate_potential_mV=float(eta[index]), candidate_j_mA_cm2=float(j[index]),
            area_cm2=area, current_multiplier=factor, normalization_basis=basis,
            image=f'{number:02d}_{candidate["sample"]}_{record["curve_id"][:10]}.png'))
    return curves


def x_limits(curve, low_current=False):
    eta, j = curve['eta'], curve['j']
    left, right = float(eta.min()), float(eta.max())
    if low_current:
        visible = eta[j >= -25]
        right = min(right, max(float(visible.max()), curve['candidate_potential_mV']) + 25)
    margin = max(5, (right - left) * .035)
    return left - margin, right + margin


def draw_curve(ax, curve, low_current=False, compact=False):
    ax.plot(curve['eta'], curve['j'], color=BLUE, linewidth=1.25 if compact else 1.65)
    ax.axhline(-10, color='#444444', linestyle='--', linewidth=.85)
    ax.axhline(-20, color='#999999', linestyle=':', linewidth=.85)
    ax.axvline(0, color='#c4c4c4', linewidth=.7)
    ax.plot(curve['candidate_potential_mV'], curve['candidate_j_mA_cm2'],
            marker='D', markersize=5 if compact else 9, markerfacecolor='white',
            markeredgecolor=ORANGE, markeredgewidth=1.4 if compact else 2, zorder=5)
    ax.set_xlim(*x_limits(curve, low_current))
    if low_current:
        ax.set_ylim(-25, 2)
    else:
        low, high = float(curve['j'].min()), max(2, float(curve['j'].max()))
        ax.set_ylim(low - max(2, (high - low) * .05), high)
    ax.grid(alpha=.22, linewidth=.6)
    ax.xaxis.set_major_locator(MaxNLocator(5 if compact else 6))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8 if compact else 12)
    if not compact:
        ax.set_xlabel('Overpotential (mV)', fontsize=13)
        ax.set_ylabel(r'Geometric current density (mA cm$^{-2}$)', fontsize=13)
        ax.set_title('Low-current view' if low_current else 'Full recorded range', fontsize=14, pad=12)


def individual_figure(curve):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3))
    fig.subplots_adjust(left=.08, right=.98, bottom=.21, top=.78, wspace=.26)
    for ax, low in zip(axes, (False, True)):
        draw_curve(ax, curve, low)
    fig.suptitle(f'{curve["number"]:02d}/35   {curve["sample"]} | {Path(curve["file"]).name}',
                 fontsize=17, fontweight='bold', y=.97)
    fig.text(.5, .89, f'Unapproved nearest-point candidate: {curve["candidate_potential_mV"]:.2f} mV'
             + rf' at {curve["candidate_j_mA_cm2"]:.2f} mA cm$^{{-2}}$',
             ha='center', fontsize=14)
    handles = [Line2D([], [], color=BLUE, lw=1.7, label='Recorded LSV'),
               Line2D([], [], color=ORANGE, marker='D', mfc='white', linestyle='none',
                      ms=7, mew=1.5, label='Nearest-point candidate'),
               Line2D([], [], color='#444444', linestyle='--', label=r'$j=-10$ mA cm$^{-2}$'),
               Line2D([], [], color='#999999', linestyle=':', label=r'$j=-20$ mA cm$^{-2}$')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .08), ncol=4,
               frameon=False, fontsize=11)
    area_note = (f'Area: {curve["area_cm2"]:g} cm2 (assumed)' if curve['normalization_basis'] == 'assumed_1cm2_unverified'
                 else f'Normalization: {curve["normalization_basis"]}')
    fig.text(.08, .045, f'Source: {curve["source_relative_path"]}', fontsize=9.5)
    fig.text(.08, .016, f'Raw TXT line: {curve["raw_line"]}    {area_note}', fontsize=9, color='#555555')
    return fig


def write_page(curves):
    index_rows, sections = [], []
    for c in curves:
        number = c['number']
        source_url = (RAW / c['file']).resolve().as_uri()
        index_rows.append(f'<tr><td><a href="#curve-{number}">{number:02d}</a></td>'
            f'<td><a href="#curve-{number}">{html.escape(c["sample"])}</a></td>'
            f'<td>{c["candidate_potential_mV"]:.2f}</td><td>{c["candidate_j_mA_cm2"]:.2f}</td>'
            f'<td><a href="{html.escape(source_url)}">{html.escape(c["source_relative_path"])}</a></td></tr>')
        previous = f'<a href="#curve-{number-1}">Previous</a>' if number > 1 else ''
        following = f'<a href="#curve-{number+1}">Next</a>' if number < len(curves) else ''
        sections.append(f'<section id="curve-{number}"><div class="section-head">'
            f'<h2>{number:02d}. {html.escape(c["sample"])}</h2><nav>{previous} '
            f'<a href="#index">Index</a> {following}</nav></div>'
            f'<a href="plots/{html.escape(c["image"])}"><img loading="lazy" width="2240" height="1008" '
            f'src="plots/{html.escape(c["image"])}" alt="{html.escape(c["sample"])} curve {number}, '
            f'candidate {c["candidate_potential_mV"]:.2f} mV at {c["candidate_j_mA_cm2"]:.2f} mA per square centimetre"></a></section>')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>35 nearest-point candidates</title><style>
*{box-sizing:border-box}body{font:16px Arial,sans-serif;color:#202020;margin:0;background:white}
main{max-width:1450px;margin:0 auto;padding:26px 24px}h1{font-size:27px;margin:0 0 8px}
h2{font-size:20px;margin:0}a{color:#156a92;text-decoration:underline;text-underline-offset:2px}
header{border-bottom:1px solid #ccc;padding-bottom:18px}header p{margin:8px 0 12px;color:#555}
nav{display:flex;gap:20px;flex-wrap:wrap}.table-wrap{overflow-x:auto;margin-top:22px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #ddd}
th{background:#f5f5f5}td:nth-child(3),td:nth-child(4){font-variant-numeric:tabular-nums;white-space:nowrap}
td:last-child{overflow-wrap:anywhere}section{padding:24px 0 16px;border-top:1px solid #bbb;scroll-margin-top:10px}
.section-head{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:8px}
img{display:block;width:100%;height:auto}.overview{margin:22px 0}
@media(max-width:650px){main{padding:18px 12px}h1{font-size:23px}.section-head{align-items:flex-start}nav{gap:12px}table{min-width:760px}}
</style></head><body><main><header><h1>Nearest-point candidates</h1>
<p>35 curves, 21 samples. Status: unapproved.</p>
<nav><a href="#index">Curve index</a><a href="overview.png">Overview</a>
<a href="nearest_point_candidates_35.pdf">35-page PDF</a><a href="plotted_candidates.csv">Point data</a></nav>
</header><div class="table-wrap" id="index"><table><thead><tr><th>No.</th><th>Sample</th>
<th>Candidate potential (mV)</th><th>j (mA cm<sup>-2</sup>)</th><th>Source TXT</th></tr></thead><tbody>'''
    page += ''.join(index_rows) + '</tbody></table></div>' + ''.join(sections) + '</main></body></html>'
    if (OUT / 'approval.json').exists():
        page = page.replace('<p>35 curves, 21 samples. Status: unapproved.</p>',
            '<p><strong>All 35 points approved on 2026-10-25 as nearest-point estimates, not eta10.</strong> '
            'The images and PDF retain their pre-approval labels. Potential and actual current density '
            'are recorded separately from eta10 and scan limits. '
            '<a href="approval.json">Approval record</a></p>')
    (OUT / 'report.html').write_text(page, encoding='utf-8')


def main():
    protected = [SOURCE, RAW / 'manifest.csv', HERE / 'eta10_curves.csv', HERE / 'eta10_samples.csv',
                 HERE / 'extraction_summary.json', HERE / 'approved_rising_points.csv',
                 HERE / 'approved_additional_estimates.csv']
    before = {str(p): digest(p) for p in protected}
    arial = font_manager.findfont('Arial', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.unicode_minus': True, 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})
    curves = load_candidates()
    (OUT / 'plots').mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT / 'nearest_point_candidates_35.pdf',
                  metadata={'Title': '35 unapproved nearest-point LSV candidates', 'CreationDate': None}) as pdf:
        for curve in curves:
            fig = individual_figure(curve)
            fig.savefig(OUT / 'plots' / curve['image'], dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    fig, axes = plt.subplots(7, 5, figsize=(17, 21))
    fig.subplots_adjust(left=.055, right=.985, top=.95, bottom=.04, wspace=.32, hspace=.52)
    for ax, curve in zip(axes.flat, curves):
        draw_curve(ax, curve, low_current=True, compact=True)
        ax.set_title(f'{curve["number"]:02d}  {curve["sample"]}\n{curve["candidate_potential_mV"]:.2f} mV; '
                     + rf'{curve["candidate_j_mA_cm2"]:.2f} mA cm$^{{-2}}$', fontsize=9.5, pad=5)
    fig.suptitle('35 unapproved nearest-point candidates', fontsize=21, y=.985)
    fig.supxlabel('Overpotential (mV); individual panel ranges', fontsize=13, y=.013)
    fig.supylabel(r'Geometric current density (mA cm$^{-2}$)', fontsize=13, x=.009)
    fig.savefig(OUT / 'overview.png', dpi=150)
    plt.close(fig)
    exported = [{k: v for k, v in curve.items() if k not in ('eta', 'j', 'index')} for curve in curves]
    write_csv(OUT / 'plotted_candidates.csv', exported)
    write_page(curves)
    after = {str(p): digest(p) for p in protected}
    assert before == after, 'A classification or approval input changed during plotting'
    summary = dict(curves=len(curves), samples=len({c['sample'] for c in curves}),
        candidate_points_verified_against_raw=True, raw_hashes_verified=True,
        classifications_and_approvals_unchanged=True, font='Arial', font_file=arial,
        normalization='Same manifest and load_curve function as the extraction',
        smoothing=False, extra_iR=False, ML_rerun=False, Figure4_changed=False,
        input_sha256=before, script_sha256=digest(Path(__file__)))
    (OUT / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not (OUT / 'approval.json').exists():
        (OUT / 'README.md').write_text(
            '# Unapproved nearest-point curve review\n\n'
            'Open report.html for the indexed 35-curve review, or the 35-page PDF.\n'
            'Each curve has a full recorded range and a low-current view with its candidate marked.\n'
            'Dashed and dotted lines show -10 and -20 mA/cm2. All figures use Arial.\n\n'
            'The points are unapproved estimates, not measured eta10 or automatically identified HER onsets.\n'
            'Potential and actual current density are reported together. No curve was approved or rejected here.\n'
            'Raw currents use the extraction manifest normalization, including flagged 1 cm2 assumptions.\n'
            'No smoothing, additional iR correction, target clipping, ML fitting or Figure 4 changes were applied.\n\n'
            'Reproduce from revision_figure with `python plot_unapproved_estimates.py`.\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'input_sha256'}, indent=2))
    print(OUT / 'report.html')


if __name__ == '__main__':
    main()
