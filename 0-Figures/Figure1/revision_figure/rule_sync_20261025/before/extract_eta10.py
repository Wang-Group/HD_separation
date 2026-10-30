"""Extract eta10 and explicitly approved, separately labelled rising points."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[2] / 'hd_separation_ml/data_extraction/all_data/LSV_repeat'
OFFSET = 0.9268
TARGET = -10.0


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def crossings(eta, j):
    result = []
    i = 0
    while i < len(j):
        if j[i] == TARGET:
            end = i
            while end + 1 < len(j) and j[end + 1] == TARGET:
                end += 1
            k = max(range(i, end + 1), key=lambda k: eta[k])
            result.append((eta[k], k, k))
            i = end + 1
            continue
        if i + 1 < len(j) and (j[i] - TARGET) * (j[i + 1] - TARGET) < 0:
            fraction = (TARGET - j[i]) / (j[i + 1] - j[i])
            result.append((eta[i] + fraction * (eta[i + 1] - eta[i]), i, i + 1))
        i += 1
    return result


def select(eta, j):
    hits = crossings(eta, j)
    if j[-1] > TARGET:
        k = max(range(len(eta)), key=lambda k: eta[k])
        return 'scan_limit', eta[k], j[k], k, k, hits
    if hits:
        value, before, after = max(hits, key=lambda p: p[0])
        return 'eta10_crossing', value, TARGET, before, after, hits
    if all(v < TARGET for v in j):
        return 'above_target_no_crossing', None, None, None, None, hits
    return 'needs_review', None, None, None, None, hits


def extract():
    manifest = read_csv(RAW / 'manifest.csv')
    approved_rows = read_csv(HERE / 'approved_rising_points.csv')
    approved = {r['curve_id']: r for r in approved_rows}
    assert len(approved) == len(approved_rows) == 20
    assert set(approved).issubset({r['curve_id'] for r in manifest})
    assert len({r['file'] for r in manifest}) == len(manifest)
    assert {p.relative_to(RAW).as_posix() for p in RAW.rglob('*.txt')} == {r['file'] for r in manifest}
    rows = []
    for r in manifest:
        path = (RAW / r['file']).resolve()
        assert path.is_relative_to(RAW.resolve())
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        assert sha == r['sha256'], r['file']
        lines = content.decode('utf-8-sig').splitlines()
        header = next(i for i, line in enumerate(lines[:50]) if
                      ('Potential(V)' in line and 'Current(A)' in line) or
                      ('E(V)' in line and 'i(A' in line))
        points = [(i + 1, *map(float, line.split())) for i, line in enumerate(lines)
                  if i > header and line.strip()]
        assert len(points) >= 2 and all(math.isfinite(v) for p in points for v in p)
        area = float(r['area_cm2']) if r['area_cm2'] else None
        if area is not None:
            assert area > 0
            factor, basis = 1000 / area, 'confirmed_area_current_override'
        elif r['exported_unit'] == 'density':
            factor, basis = 1000, 'exported_density_no_area_division'
        elif r['exported_unit'] == 'current':
            area, factor, basis = 1.0, 1000, 'assumed_1cm2_unverified'
        else:
            raise ValueError('Unknown unit: ' + r['file'])
        eta = [-1000 * (p[1] + OFFSET) for p in points]
        j = [p[2] * factor for p in points]
        status, value, current, before, after, hits = select(eta, j)
        if r['curve_id'] in approved:
            decision = approved[r['curve_id']]
            assert decision['sha256'] == sha
            index = next(i for i, point in enumerate(points) if point[0] == int(decision['raw_line']))
            assert status != 'scan_limit', 'Approved point conflicts with terminal rule'
            status, value, current, before, after = 'rising_point', eta[index], j[index], index, index
            assert abs(value - float(decision['expected_potential_mV'])) < 1e-6
            assert abs(current - float(decision['expected_j_mA_cm2'])) < 1e-6
        rows.append(dict(sample=r['sample_group'], file=r['file'], status=status,
            eta10_mV=value if status == 'eta10_crossing' else None,
            rising_point_potential_mV=value if status == 'rising_point' else None,
            rising_point_j_mA_cm2=current if status == 'rising_point' else None,
            scan_limit_mV=value if status == 'scan_limit' else None,
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
            source_relative_path=r['source_relative_path'], sha256=sha, curve_id=r['curve_id']))
    rows.sort(key=lambda r: (r['sample'], r['file']))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['sample']].append(r)
    summary = []
    for sample, group in sorted(grouped.items()):
        values = [r['eta10_mV'] for r in group if r['eta10_mV'] is not None]
        rising = [r['rising_point_potential_mV'] for r in group if r['rising_point_potential_mV'] is not None]
        rising_j = [r['rising_point_j_mA_cm2'] for r in group if r['rising_point_j_mA_cm2'] is not None]
        counts = Counter(r['status'] for r in group)
        summary.append(dict(sample=sample, raw_curves=len(group), n_eta10=len(values),
            eta10_mean_mV=statistics.mean(values) if values else None,
            eta10_SD_mV=statistics.stdev(values) if len(values) > 1 else None,
            n_rising_points=len(rising),
            rising_point_mean_mV=statistics.mean(rising) if rising else None,
            rising_point_SD_mV=statistics.stdev(rising) if len(rising) > 1 else None,
            rising_point_j_mean_mA_cm2=statistics.mean(rising_j) if rising_j else None,
            rising_point_j_SD_mA_cm2=statistics.stdev(rising_j) if len(rising_j) > 1 else None,
            n_scan_limit=counts['scan_limit'], n_above_target_no_crossing=counts['above_target_no_crossing'],
            n_needs_review=counts['needs_review']))
    write_csv(HERE / 'eta10_curves.csv', rows)
    write_csv(HERE / 'eta10_samples.csv', summary)
    report = dict(raw_files=len(rows), samples=len(summary), counts=dict(Counter(r['status'] for r in rows)),
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
        value = next((r[k] for k in ('eta10_mV', 'rising_point_potential_mV', 'scan_limit_mV') if r[k] is not None), None)
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
