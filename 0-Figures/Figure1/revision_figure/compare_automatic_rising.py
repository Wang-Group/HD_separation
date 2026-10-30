"""Compare one approval-independent rising-point rule against the current extraction."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import scipy

from extract_eta10 import HERE, RAW, load_curve, read_csv, write_csv
from lsv_rules import CONFIG, automatic_selection

OUT = HERE / 'automatic_rising_review_20261025'
METRICS = {
    'eta10_crossing': ('eta10_mV', None),
    'rising_point': ('rising_point_potential_mV', 'rising_point_j_mA_cm2'),
    'nearest_point_estimate': ('nearest_point_potential_mV', 'nearest_point_j_mA_cm2'),
    'scan_limit': ('scan_limit_mV', None),
}
PROTECTED_DIRS = ('ml_revision', '0-Figures/Figure4',
    'hd_separation_ml/data_extraction/all_data/LSV_repeat',
    'hd_separation_ml/data_extraction/all_data/LSV',
    '0-Figures/Figure1/revision_figure/figure1b_outputs')
PROTECTED_FILES = ('eta10_curves.csv', 'eta10_samples.csv', 'extraction_summary.json',
    'approved_rising_points.csv', 'approved_additional_estimates.csv', 'extract_eta10.py',
    'reclassify_nearest_as_rising.py', 'lsv_rules.py', 'lsv_noise_exceptions.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes():
    result = {'files': {name: digest(HERE / name) for name in PROTECTED_FILES}, 'directories': {}}
    for relative in PROTECTED_DIRS:
        root = HERE.parents[2] / relative
        paths = sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        checksum = hashlib.sha256()
        for path in paths:
            checksum.update(path.relative_to(root).as_posix().encode('utf-8'))
            checksum.update(hashlib.sha256(path.read_bytes()).digest())
        result['directories'][relative] = dict(files=len(paths), sha256=checksum.hexdigest())
    return result


def baseline_value(row):
    field = METRICS.get(row['status'], (None, None))[0]
    return float(row[field]) if field and row[field] != '' else None


def differs(a, b):
    if a in ('', None) or b in ('', None):
        return (a in ('', None)) != (b in ('', None))
    return abs(float(a) - float(b)) > 1e-6


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['sample']].append(row)
    summaries = {}
    for sample, group in groups.items():
        out = dict(sample=sample, raw_curves=len(group))
        for status, prefix in (('eta10_crossing', 'eta10'), ('rising_point', 'rising_point'),
                               ('nearest_point_estimate', 'nearest_point')):
            values = [baseline_value(r) for r in group if r['status'] == status]
            out[f'n_{prefix}'] = len(values)
            out[f'{prefix}_mean_mV'] = statistics.mean(values) if values else None
            out[f'{prefix}_SD_mV'] = statistics.stdev(values) if len(values) > 1 else None
        out['n_scan_limit'] = sum(r['status'] == 'scan_limit' for r in group)
        out['n_no_accepted_result'] = sum(r['status'] == 'above_target_no_crossing' for r in group)
        summaries[sample] = out
    return summaries


def run():
    protected_before = protected_hashes()
    assert digest(OUT / 'before/eta10_curves.csv') == '243da67324e92d4cd49d911c7c04b8b62f51b6d200a10ea410eb71f624ac6e13'
    previous = {r['curve_id']: r for r in read_csv(OUT / 'before/eta10_curves.csv')}
    manifest = read_csv(RAW / 'manifest.csv')
    assert len(manifest) == len(previous) == len({r['curve_id'] for r in manifest})
    proposed, comparisons, candidate_rows = [], [], []
    for record in manifest:
        points, eta, j, area, factor, basis = load_curve(record)
        old = previous[record['curve_id']]
        assert old['sha256'] == record['sha256']
        decision = automatic_selection(eta, j)
        status = decision['status']
        row = dict(sample=old['sample'], file=record['file'], curve_id=record['curve_id'],
            sha256=record['sha256'], source_relative_path=record['source_relative_path'], origin=old['origin'],
            status=status, selected_potential_mV=decision['potential_mV'], selected_j_mA_cm2=decision['j_mA_cm2'],
            raw_line_before=points[decision['before']][0] if decision['before'] is not None else None,
            raw_line_after=points[decision['after']][0] if decision['after'] is not None else None,
            base_status=decision['base_status'], largest_crossing_mV=decision['largest_crossing_mV'],
            crossing_count=decision['crossing_count'], final_j_mA_cm2=j[-1], area_cm2=area,
            current_multiplier=factor, normalization_basis=basis,
            detector_diagnostic=decision['detector_diagnostic'],
            detected_candidates=len(decision['candidates']),
            eligible_candidates=sum(c['eligible'] for c in decision['candidates']))
        for metric_status, (potential_field, current_field) in METRICS.items():
            row[potential_field] = decision['potential_mV'] if status == metric_status else None
            if current_field:
                row[current_field] = decision['j_mA_cm2'] if status == metric_status else None
        proposed.append(row)
        for c in decision['candidates']:
            candidate_rows.append(dict(sample=row['sample'], file=row['file'], curve_id=row['curve_id'],
                raw_line=points[c['index']][0], **c))
        old_value, new_value = baseline_value(old), decision['potential_mV']
        changed_metric = old['status'] != status
        changed_point = differs(old_value, new_value) or differs(old['selected_j_mA_cm2'], decision['j_mA_cm2'])
        comparisons.append(dict(sample=row['sample'], file=row['file'], curve_id=row['curve_id'],
            origin=row['origin'], source_relative_path=row['source_relative_path'], sha256=row['sha256'],
            comparison='metric_changed' if changed_metric else 'point_changed' if changed_point else 'unchanged',
            current_status=old['status'], proposed_status=status, current_potential_mV=old_value,
            proposed_potential_mV=new_value,
            potential_difference_mV=new_value - old_value if new_value is not None and old_value is not None else None,
            current_j_mA_cm2=old['selected_j_mA_cm2'], proposed_j_mA_cm2=decision['j_mA_cm2'],
            current_raw_line=old['raw_line_before'], proposed_raw_line=row['raw_line_before'],
            current_raw_line_after=old['raw_line_after'], proposed_raw_line_after=row['raw_line_after'],
            largest_crossing_mV=decision['largest_crossing_mV'], eligible_candidates=row['eligible_candidates'],
            detector_diagnostic=row['detector_diagnostic']))
    for rows in (proposed, comparisons):
        rows.sort(key=lambda r: (r['sample'], r['file']))
    changed = [r for r in comparisons if r['comparison'] != 'unchanged']
    write_csv(OUT / 'proposed_curves.csv', proposed)
    write_csv(OUT / 'curve_comparison.csv', comparisons)
    write_csv(OUT / 'changed_curves.csv', changed, fieldnames=list(comparisons[0]))
    write_csv(OUT / 'all_detected_candidates.csv', candidate_rows)
    old_samples, new_samples = summarize(previous.values()), summarize(proposed)
    assert old_samples.keys() == new_samples.keys()
    sample_comparisons = []
    for sample in sorted(old_samples):
        old, new = old_samples[sample], new_samples[sample]
        changes = [key for key in old if key != 'sample' and differs(old[key], new[key])]
        if not changes:
            continue
        result = dict(sample=sample, changed_statistics=';'.join(changes))
        for key in old:
            if key != 'sample':
                result['current_' + key] = old[key]
                result['proposed_' + key] = new[key]
        sample_comparisons.append(result)
    write_csv(OUT / 'proposed_samples.csv', [new_samples[k] for k in sorted(new_samples)])
    write_csv(OUT / 'changed_samples.csv', sample_comparisons)
    counts_before = dict(Counter(r['status'] for r in previous.values()))
    counts_after = dict(Counter(r['status'] for r in proposed))
    report = dict(proposal_only=True, adopted=False, selection_uses_approval_list=False,
        raw_curves=len(proposed), counts_current=counts_before, counts_proposed=counts_after,
        counts_proposed_by_origin={origin: dict(Counter(r['status'] for r in proposed if r['origin'] == origin))
            for origin in ('original', 'additional')},
        changed_curves=len(changed), affected_sample_labels=len({r['sample'] for r in changed}),
        changed_samples=len(sample_comparisons),
        transitions=dict(Counter(r['current_status'] + ' -> ' + r['proposed_status'] for r in changed)),
        detector_diagnostics=dict(Counter(r['detector_diagnostic'] for r in proposed)),
        sample_coverage_current=dict(eta10=sum(r['n_eta10'] > 0 for r in old_samples.values()),
            eta10_with_SD=sum(r['n_eta10'] > 1 for r in old_samples.values())),
        sample_coverage_proposed=dict(eta10=sum(r['n_eta10'] > 0 for r in new_samples.values()),
            eta10_with_SD=sum(r['n_eta10'] > 1 for r in new_samples.values())),
        detector_config=CONFIG, selected_raw_current_window='-20 < j < -10 mA/cm2',
        require_after_largest_crossing=True, choose_largest_eligible_overpotential=True,
        smoothing_only_for_detection=True, no_automatic_HER_attribution=True,
        seed_required=False, numpy_version=np.__version__, scipy_version=scipy.__version__,
        protected_before=protected_before, protected_files_and_directories_unchanged=True,
        script_sha256=digest(Path(__file__)), detector_script_sha256=digest(HERE / 'lsv_rules.py'),
        ML_rerun=False, manuscript_updated=False, publication_figures_updated=False)
    assert protected_before == protected_hashes()
    (OUT / 'comparison_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('counts_current', 'counts_proposed', 'changed_curves',
        'affected_sample_labels', 'sample_coverage_current', 'sample_coverage_proposed', 'transitions')}, indent=2))
    return changed


def plot(changed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    import textwrap
    font_manager.findfont('Arial', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11, 'pdf.fonttype': 42,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold'})
    records = {r['curve_id']: r for r in read_csv(RAW / 'manifest.csv')}
    curves = []
    for row in changed:
        points, eta, j, area, _, basis = load_curve(records[row['curve_id']])
        curves.append(dict(row, eta=np.asarray(eta), j=np.asarray(j), area=area, basis=basis))

    def draw(ax, curve, local=False):
        x, y = curve['eta'], curve['j']
        ax.plot(x, y, color='#176b9a', lw=1.2)
        ax.axhline(-10, color='#444444', ls='--', lw=.8)
        ax.axhline(-20, color='#999999', ls=':', lw=.8)
        old_x, new_x = float(curve['current_potential_mV']), float(curve['proposed_potential_mV'])
        old_y, new_y = float(curve['current_j_mA_cm2']), float(curve['proposed_j_mA_cm2'])
        ax.plot(old_x, old_y, marker='s', mfc='none', mec='#a23b72', mew=1.6, ms=10, zorder=5)
        ax.plot(new_x, new_y, marker='D', mfc='none', mec='#008879', mew=1.6, ms=7, zorder=6)
        if local:
            ax.set_xlim(max(float(x.min()) - 5, min(old_x, new_x) - 60),
                        min(float(x.max()) + 5, max(old_x, new_x) + 70))
            ax.set_ylim(-25, 3)
        else:
            ax.set_xlim(float(x.min()) - 10, float(x.max()) + 10)
            ax.set_ylim(min(-25, float(y.min()) - 5), max(3, float(y.max()) + 2))
        ax.grid(alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('Overpotential (mV)')
        ax.set_ylabel(r'Current density (mA cm$^{-2}$)')

    handles = [Line2D([], [], color='#176b9a', label='Recorded LSV'),
        Line2D([], [], marker='s', mfc='none', mec='#a23b72', linestyle='none', ms=9, label='Current selection'),
        Line2D([], [], marker='D', mfc='none', mec='#008879', linestyle='none', ms=7, label='Automatic proposal'),
        Line2D([], [], color='#444444', ls='--', label=r'$j=-10$ mA cm$^{-2}$'),
        Line2D([], [], color='#999999', ls=':', label=r'$j=-20$ mA cm$^{-2}$')]
    with PdfPages(OUT / 'changed_curves_review.pdf', metadata={'Title': 'Automatic rising-point rule: proposed changes',
                                                            'CreationDate': None}) as pdf:
        for number, curve in enumerate(curves, 1):
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            fig.subplots_adjust(left=.075, right=.98, bottom=.22, top=.72, wspace=.27)
            for ax, local in zip(axes, (False, True)):
                draw(ax, curve, local)
                ax.set_title('Selection detail' if local else 'Full recorded range', pad=10)
            fig.suptitle(f'{number}/{len(curves)}  {curve["sample"]} | {Path(curve["file"]).name}',
                         fontsize=16, fontweight='bold', y=.97)
            old_type = r'$\eta_{10}$ crossing' if curve['current_status'] == 'eta10_crossing' else 'rising-point estimate'
            fig.text(.5, .90, f'Current {old_type}: {float(curve["current_potential_mV"]):.2f} mV at '
                + rf'{float(curve["current_j_mA_cm2"]):.2f} mA cm$^{{-2}}$', ha='center', color='#a23b72', fontsize=12)
            fig.text(.5, .845, f'Proposed rising-point estimate: {float(curve["proposed_potential_mV"]):.2f} mV at '
                + rf'{float(curve["proposed_j_mA_cm2"]):.2f} mA cm$^{{-2}}$', ha='center', color='#008879', fontsize=12)
            fig.text(.5, .792, r'Proposal only. Rising potential/current remain separate from measured $\eta_{10}$.',
                     ha='center', fontsize=10, color='#555555')
            fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .10), ncol=5, frameon=False, fontsize=10)
            source = textwrap.fill('Source: ' + curve['source_relative_path'], width=155)
            fig.text(.075, .09, source, va='top', fontsize=8.5)
            fig.text(.075, .023, f'Current raw bracket: {curve["current_raw_line"]}-{curve["current_raw_line_after"]}; proposed raw line: '
                f'{curve["proposed_raw_line"]}. Normalization: {curve["basis"]}.', fontsize=8.5, color='#555555')
            pdf.savefig(fig)
            plt.close(fig)
    nrows = max(1, (len(curves) + 1) // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, nrows * 3.5 + 1.5), squeeze=False)
    fig.subplots_adjust(left=.075, right=.98, bottom=.09, top=.91, wspace=.24, hspace=.52)
    for ax, curve in zip(axes.flat, curves):
        draw(ax, curve, True)
        ax.set_title(f'{curve["sample"]} | {Path(curve["file"]).name}\n'
            f'{float(curve["current_potential_mV"]):.2f} -> {float(curve["proposed_potential_mV"]):.2f} mV', fontsize=11)
    for ax in list(axes.flat)[len(curves):]:
        ax.set_visible(False)
    fig.suptitle('Automatic rising-point rule | Proposed changes only', fontsize=17, y=.98)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .015), ncol=5, frameon=False)
    fig.savefig(OUT / 'changed_curves_overview.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots', action='store_true', help='Also export the changed-curve PDF and overview PNG')
    args = parser.parse_args()
    changed = run()
    if args.plots:
        plot(changed)
