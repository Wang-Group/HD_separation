"""Extract eta10 and automatic point estimates, retaining reviewed noise exceptions."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from lsv_rules import CONFIG, automatic_selection, apply_noise_exception, crossings, select

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[2] / 'hd_separation_ml/data_extraction/all_data/LSV_repeat'
# Measured electrolyte pH: 13.9 +/- 0.1, consistent with nominal pH = 14.
# Hg/HgO-to-RHE processing uses 0.098 + 0.0592 * 14 = 0.9268 V.
# Keep this fixed convention for both initial and repeated LSV curves.
OFFSET = 0.9268
TARGET = -10.0
ESTIMATE_MIN_J = -20.0
RULE_VERSION = '2026-10-25-automatic-rising-with-noise-review'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames if fieldnames is not None else list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_curve(record):
    path = (RAW / record['file']).resolve()
    assert path.is_relative_to(RAW.resolve())
    content = path.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    assert sha == record['sha256'], record['file']
    lines = content.decode('utf-8-sig').splitlines()
    header = next(i for i, line in enumerate(lines[:50]) if
                  ('Potential(V)' in line and 'Current(A)' in line) or
                  ('E(V)' in line and 'i(A' in line))
    points = [(i + 1, *map(float, line.split())) for i, line in enumerate(lines)
              if i > header and line.strip()]
    assert len(points) >= 2 and all(math.isfinite(v) for p in points for v in p)
    area = float(record['area_cm2']) if record['area_cm2'] else None
    if area is not None:
        assert area > 0
        factor, basis = 1000 / area, 'confirmed_area_current_override'
    elif record['exported_unit'] == 'density':
        factor, basis = 1000, 'exported_density_no_area_division'
    elif record['exported_unit'] == 'current':
        area, factor, basis = 1.0, 1000, 'assumed_1cm2_unverified'
    else:
        raise ValueError('Unknown unit: ' + record['file'])
    eta = [-1000 * (p[1] + OFFSET) for p in points]
    j = [p[2] * factor for p in points]
    return points, eta, j, area, factor, basis


def approved_selection(eta, j, points, decision):
    status, _, _, _, _, hits = select(eta, j)
    assert status != 'scan_limit', 'Approved point conflicts with terminal rule'
    index = next(i for i, point in enumerate(points) if point[0] == int(decision['raw_line']))
    kind = decision['estimate_type']
    assert kind in ('rising_point', 'nearest_point_estimate')
    assert ESTIMATE_MIN_J < j[index] < TARGET, 'Estimate must satisfy -20 < j < -10 before rounding'
    if kind == 'nearest_point_estimate':
        assert status == 'above_target_no_crossing', 'Nearest point cannot replace a usable crossing'
        assert abs(abs(j[index] - TARGET) - min(abs(v - TARGET) for v in j)) < 1e-9
    assert abs(eta[index] - float(decision['expected_potential_mV'])) < 1e-6
    assert abs(j[index] - float(decision['expected_j_mA_cm2'])) < 1e-6
    return kind, eta[index], j[index], index, index, hits


def load_approvals():
    """Read the superseded approval register for historical audit tests only."""
    rising = read_csv(HERE / 'approved_rising_points.csv')
    assert len(rising) == 20
    for row in rising:
        row.update(estimate_type='rising_point', approval_basis='reviewed_LSV_points_20261023')
    rows = rising + read_csv(HERE / 'approved_additional_estimates.csv')
    approved = {r['curve_id']: r for r in rows}
    assert len(approved) == len(rows), 'An estimate may have only one adopted selection'
    return approved


def load_noise_exceptions():
    data = json.loads((HERE / 'lsv_noise_exceptions.json').read_text(encoding='utf-8'))
    records = {r['curve_id']: r for r in data['exceptions']}
    assert len(records) == len(data['exceptions'])
    return records


def extract():
    manifest = read_csv(RAW / 'manifest.csv')
    exceptions = load_noise_exceptions()
    assert set(exceptions).issubset({r['curve_id'] for r in manifest})
    assert len({r['file'] for r in manifest}) == len(manifest)
    assert {p.relative_to(RAW).as_posix() for p in RAW.rglob('*.txt')} == {r['file'] for r in manifest}
    rows = []
    for r in manifest:
        points, eta, j, area, factor, basis = load_curve(r)
        sha = r['sha256']
        automatic = automatic_selection(eta, j)
        exception = exceptions.get(r['curve_id'])
        decision = automatic
        if exception:
            assert exception['sha256'] == sha and exception['file'] == r['file']
            assert exception['source_relative_path'] == r['source_relative_path']
            assert points[automatic['before']][0] == exception['rejected_automatic_raw_line']
            decision = apply_noise_exception(automatic, eta, j, exception)
            assert points[decision['before']][0] == exception['retained_raw_line_before']
            assert points[decision['after']][0] == exception['retained_raw_line_after']
        status, value, current = decision['status'], decision['potential_mV'], decision['j_mA_cm2']
        before, after = decision['before'], decision['after']
        base_status, hits = decision['base_status'], crossings(eta, j)
        is_estimate = status in ('rising_point', 'nearest_point_estimate')
        selection_basis = 'reviewed_noise_exception_keep_crossing' if exception else {
            'scan_limit': 'terminal_current_scan_limit', 'eta10_crossing': 'largest_interpolated_crossing',
            'rising_point': 'automatic_largest_eligible_rising_point',
            'nearest_point_estimate': 'automatic_nearest_point_fallback',
            'above_target_no_crossing': 'no_eligible_crossing_or_estimate',
            'needs_review': 'unclassified_review_required'}[status]
        nearest = min(range(len(j)), key=lambda k: (abs(j[k] - TARGET), -eta[k]))
        rows.append(dict(sample=r['sample_group'], file=r['file'], status=status,
            eta10_mV=value if status == 'eta10_crossing' else None,
            rising_point_potential_mV=value if status == 'rising_point' else None,
            rising_point_j_mA_cm2=current if status == 'rising_point' else None,
            nearest_point_potential_mV=value if status == 'nearest_point_estimate' else None,
            nearest_point_j_mA_cm2=current if status == 'nearest_point_estimate' else None,
            estimate_type=status if is_estimate else '',
            estimate_potential_mV=value if is_estimate else None,
            estimate_j_mA_cm2=current if is_estimate else None,
            selection_basis=selection_basis,
            approved_estimate_potential_mV=value if is_estimate else None,
            approved_estimate_j_mA_cm2=current if is_estimate else None,
            approval_basis=selection_basis if is_estimate else '',
            noise_exception_id=exception['exception_id'] if exception else '',
            noise_exception_reason=exception['reason'] if exception else '',
            automatic_status=automatic['status'], automatic_potential_mV=automatic['potential_mV'],
            automatic_j_mA_cm2=automatic['j_mA_cm2'],
            automatic_raw_line_before=points[automatic['before']][0] if automatic['before'] is not None else None,
            detector_diagnostic=automatic['detector_diagnostic'],
            detected_rising_candidates=len(automatic['candidates']),
            eligible_rising_candidates=sum(c['eligible'] for c in automatic['candidates']),
            scan_limit_mV=value if status == 'scan_limit' else None,
            base_status=base_status,
            scan_limit_reason=('final_below_target_after_crossing' if hits else 'never_reaches_target')
                if status == 'scan_limit' else '',
            nearest_candidate_potential_mV=eta[nearest], nearest_candidate_j_mA_cm2=j[nearest],
            nearest_candidate_within_estimate_window=ESTIMATE_MIN_J < j[nearest] < TARGET,
            selected_j_mA_cm2=current, crossing_count=len(hits),
            largest_crossing_mV=max((h[0] for h in hits), default=None),
            final_j_mA_cm2=j[-1], final_eta_mV=eta[-1], min_eta_mV=min(eta), max_eta_mV=max(eta),
            max_cathodic_magnitude_mA_cm2=max(-v for v in j),
            area_cm2=area, normalization_basis=basis, current_multiplier=factor,
            exported_unit=r['exported_unit'], RHE_offset_V=OFFSET, extra_iR_applied=False,
            raw_line_before=points[before][0] if before is not None else None,
            raw_line_after=points[after][0] if after is not None else None,
            E_before_V=points[before][1] if before is not None else None,
            E_after_V=points[after][1] if after is not None else None,
            j_before_mA_cm2=j[before] if before is not None else None,
            j_after_mA_cm2=j[after] if after is not None else None,
            scan_reversal=any(eta[i+1] < eta[i]-1 for i in range(len(eta)-1)),
            source_relative_path=r['source_relative_path'], sha256=sha, curve_id=r['curve_id'],
            origin='original' if r['source_relative_path'].startswith('LSV/') else 'additional',
            source_family=r['source_relative_path'].split('/')[0], rule_version=RULE_VERSION))
    rows.sort(key=lambda r: (r['sample'], r['file']))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['sample']].append(r)
    summary = []
    for sample, group in sorted(grouped.items()):
        values = [r['eta10_mV'] for r in group if r['eta10_mV'] is not None]
        rising = [r['rising_point_potential_mV'] for r in group if r['rising_point_potential_mV'] is not None]
        rising_j = [r['rising_point_j_mA_cm2'] for r in group if r['rising_point_j_mA_cm2'] is not None]
        nearest = [r['nearest_point_potential_mV'] for r in group if r['nearest_point_potential_mV'] is not None]
        nearest_j = [r['nearest_point_j_mA_cm2'] for r in group if r['nearest_point_j_mA_cm2'] is not None]
        counts = Counter(r['status'] for r in group)
        summary.append(dict(sample=sample, raw_curves=len(group), n_eta10=len(values),
            eta10_mean_mV=statistics.mean(values) if values else None,
            eta10_SD_mV=statistics.stdev(values) if len(values) > 1 else None,
            n_rising_points=len(rising),
            rising_point_mean_mV=statistics.mean(rising) if rising else None,
            rising_point_SD_mV=statistics.stdev(rising) if len(rising) > 1 else None,
            rising_point_j_mean_mA_cm2=statistics.mean(rising_j) if rising_j else None,
            rising_point_j_SD_mA_cm2=statistics.stdev(rising_j) if len(rising_j) > 1 else None,
            n_nearest_point_estimates=len(nearest), n_estimates=len(rising) + len(nearest),
            n_approved_estimates=len(rising) + len(nearest),
            nearest_point_mean_mV=statistics.mean(nearest) if nearest else None,
            nearest_point_SD_mV=statistics.stdev(nearest) if len(nearest) > 1 else None,
            nearest_point_j_mean_mA_cm2=statistics.mean(nearest_j) if nearest_j else None,
            nearest_point_j_SD_mA_cm2=statistics.stdev(nearest_j) if len(nearest_j) > 1 else None,
            n_scan_limit=counts['scan_limit'], n_above_target_no_crossing=counts['above_target_no_crossing'],
            n_needs_review=counts['needs_review']))
    write_csv(HERE / 'eta10_curves.csv', rows)
    write_csv(HERE / 'eta10_samples.csv', summary)
    report = dict(raw_files=len(rows), samples=len(summary), counts=dict(Counter(r['status'] for r in rows)),
                  counts_by_origin={origin: dict(Counter(r['status'] for r in rows if r['origin'] == origin))
                                    for origin in ('original', 'additional')},
                  estimate_count=sum(s['n_estimates'] for s in summary),
                  approved_estimates=sum(s['n_estimates'] for s in summary),
                  selection_uses_approval_list=False, explicit_noise_exceptions=len(exceptions),
                  detector_config=CONFIG,
                  unapproved_nearest_candidates_within_window=sum(
                      r['status'] == 'above_target_no_crossing' and r['nearest_candidate_within_estimate_window']
                      for r in rows),
                  sample_coverage=dict(accepted_eta10=sum(s['n_eta10'] > 0 for s in summary),
                      eta10_with_SD=sum(s['n_eta10'] > 1 for s in summary),
                      any_approved_estimate=sum(s['n_approved_estimates'] > 0 for s in summary),
                      estimate_without_accepted_eta10=sum(s['n_approved_estimates'] > 0 and s['n_eta10'] == 0 for s in summary),
                      any_scan_limit=sum(s['n_scan_limit'] > 0 for s in summary)),
                  rule_version=RULE_VERSION, estimate_window='-20 < raw selected j < -10 mA/cm2; automatic selection with explicit noise exceptions',
                  terminal_gate_retained=True, target_capping_applied=False,
                  input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [RAW / 'manifest.csv', HERE / 'lsv_noise_exceptions.json', HERE / 'lsv_rules.py']},
                  legacy_approved_fields='Compatibility aliases for adopted estimates; not evidence of individual manual approval.',
                  figures_refreshed=False, ML_rerun=False,
                  hashes_verified=True, raw_modified=False, extra_iR_applied=False)
    (HERE / 'extraction_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return rows, report


def compare(rows, baseline):
    old_rows = read_csv(baseline)
    old = {r['curve_id']: r for r in old_rows}
    assert len(old) == len(old_rows)
    result = []
    for r in rows:
        previous = old.get(r['curve_id'])
        value = next((r[k] for k in ('eta10_mV', 'rising_point_potential_mV',
                                    'nearest_point_potential_mV', 'scan_limit_mV') if r[k] is not None), None)
        prior = float(previous['potential_mV']) if previous else None
        kind = previous['metric_type'] if previous else 'not_in_afternoon_metric_table'
        if previous:
            assert previous['sha256'] == r['sha256']
        if not previous:
            outcome = 'unassigned_in_both' if value is None else 'newly_assigned'
        elif kind != r['status']:
            outcome = 'metric_changed'
        elif value is None:
            outcome = 'now_unassigned'
        else:
            outcome = 'match' if abs(value-prior) <= 1e-6 else 'value_changed'
        result.append(dict(sample=r['sample'], file=r['file'], current_metric=r['status'],
            current_value_mV=value, afternoon_metric=kind, afternoon_value_mV=prior,
            difference_mV=value-prior if value is not None and prior is not None else None,
            comparison=outcome, afternoon_current_mA_cm2=float(previous['current_mA_cm2']) if previous else None,
            curve_id=r['curve_id'], sha256=r['sha256']))
    folder = HERE / 'tmp_comparison'
    folder.mkdir(exist_ok=True)
    write_csv(folder / 'afternoon_metric_snapshot.csv', old_rows)
    write_csv(folder / 'eta10_vs_afternoon.csv', result)
    audit = dict(counts=dict(Counter(r['comparison'] for r in result)),
                 unmatched_afternoon_ids=sorted(set(old)-{r['curve_id'] for r in rows}),
                 baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
    (folder / 'comparison_summary.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare-afternoon', type=Path)
    args = parser.parse_args()
    rows, report = extract()
    print(json.dumps(report, indent=2))
    if args.compare_afternoon:
        compare(rows, args.compare_afternoon)
