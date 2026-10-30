"""Apply the reviewed largest-overpotential rising-point rule to the frozen 46 curves."""
import hashlib
import json

from extract_eta10 import HERE, RAW, load_curve, read_csv, write_csv
from lsv_rules import CONFIG, rising_candidates, choose_latest

REVIEW = HERE / 'rule_sync_20261025/rising_reclassification'


def prepare():
    receipt = json.loads((REVIEW / 'approval.json').read_text(encoding='utf-8'))
    baseline = REVIEW / 'before/eta10_curves.csv'
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == receipt['baseline_curves_sha256']
    assert receipt['eta10_field_must_remain_blank'] and not receipt['pool_with_eta10']
    manifest = {r['curve_id']: r for r in read_csv(RAW / 'manifest.csv')}
    rows = read_csv(baseline)
    nearest = [r for r in rows if r['status'] == 'nearest_point_estimate']
    assert len(nearest) == 46
    decisions, candidates, audit = [], [], []
    for row in nearest:
        points, eta, j, _, _, _ = load_curve(manifest[row['curve_id']])
        assert manifest[row['curve_id']]['sha256'] == row['sha256']
        turns = rising_candidates(eta, j)
        chosen = choose_latest(turns)
        for candidate in turns:
            candidates.append(dict(sample=row['sample'], file=row['file'], curve_id=row['curve_id'],
                raw_line=points[candidate['index']][0], **candidate,
                within_current_window=-20 < candidate['j_mA_cm2'] < -10,
                selected=candidate is chosen))
        if chosen is not None:
            decisions.append(dict(sample=row['sample'], file=row['file'], curve_id=row['curve_id'],
                sha256=row['sha256'], source_relative_path=row['source_relative_path'],
                previous_raw_line=row['raw_line_before'],
                previous_potential_mV=row['nearest_point_potential_mV'],
                previous_j_mA_cm2=row['nearest_point_j_mA_cm2'],
                raw_line=points[chosen['index']][0], expected_potential_mV=chosen['potential_mV'],
                expected_j_mA_cm2=chosen['j_mA_cm2'], estimate_type='rising_point',
                approval_basis='largest_eligible_rising_point_review_20261025'))
        audit.append(dict(sample=row['sample'], file=row['file'], curve_id=row['curve_id'],
            eligible_candidates=sum(-20 < c['j_mA_cm2'] < -10 for c in turns),
            old_status='nearest_point_estimate', new_status='rising_point' if chosen else 'nearest_point_estimate',
            previous_potential_mV=row['nearest_point_potential_mV'],
            previous_j_mA_cm2=row['nearest_point_j_mA_cm2'],
            selected_potential_mV=chosen['potential_mV'] if chosen else row['nearest_point_potential_mV'],
            selected_j_mA_cm2=chosen['j_mA_cm2'] if chosen else row['nearest_point_j_mA_cm2']))
    # Existing explicitly reviewed rising points remain adopted; check for later alternatives.
    existing = [r for r in rows if r['status'] == 'rising_point']
    assert len(existing) == 24
    for row in existing:
        _, eta, j, _, _, _ = load_curve(manifest[row['curve_id']])
        latest = choose_latest(rising_candidates(eta, j))
        assert latest is None or latest['potential_mV'] <= float(row['rising_point_potential_mV']) + 1e-6
    assert len(decisions) == 39
    write_csv(REVIEW / 'approved_reassignments.csv', decisions)
    write_csv(REVIEW / 'all_detected_candidates.csv', candidates)
    write_csv(REVIEW / 'nearest_46_audit.csv', audit)
    summary = dict(reviewed_nearest_curves=46, reclassified_as_rising=len(decisions),
        retained_nearest=46 - len(decisions), existing_rising_checked=len(existing),
        existing_rising_values_unchanged=True, selection='largest potential among eligible candidates',
        selected_current_window='-20 < raw j < -10 mA/cm2', detection_config=CONFIG,
        smoothing_only_for_candidate_detection=True, adopted_values_from_raw=True,
        approval_sha256=hashlib.sha256((REVIEW / 'approval.json').read_bytes()).hexdigest())
    (REVIEW / 'candidate_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


def verify():
    before = {r['curve_id']: r for r in read_csv(REVIEW / 'before/eta10_curves.csv')}
    after = {r['curve_id']: r for r in read_csv(HERE / 'eta10_curves.csv')}
    decisions = {r['curve_id']: r for r in read_csv(REVIEW / 'approved_reassignments.csv')}
    assert before.keys() == after.keys()
    changed = {key for key in before if before[key] != after[key]}
    assert changed == decisions.keys() and len(changed) == 39
    for key, decision in decisions.items():
        old, new = before[key], after[key]
        assert old['status'] == 'nearest_point_estimate' and new['status'] == 'rising_point'
        assert old['sha256'] == new['sha256'] == decision['sha256']
        assert new['eta10_mV'] == new['scan_limit_mV'] == ''
        assert new['nearest_point_potential_mV'] == new['nearest_point_j_mA_cm2'] == ''
        assert new['raw_line_before'] == decision['raw_line']
        assert float(new['rising_point_potential_mV']) == float(decision['expected_potential_mV'])
        assert float(new['rising_point_j_mA_cm2']) == float(decision['expected_j_mA_cm2'])
    old_samples = {r['sample']: r for r in read_csv(REVIEW / 'before/eta10_samples.csv')}
    new_samples = {r['sample']: r for r in read_csv(HERE / 'eta10_samples.csv')}
    assert old_samples.keys() == new_samples.keys()
    unchanged_fields = ('raw_curves', 'n_eta10', 'eta10_mean_mV', 'eta10_SD_mV',
        'n_approved_estimates', 'n_scan_limit', 'n_above_target_no_crossing', 'n_needs_review')
    for sample in old_samples:
        assert all(old_samples[sample][key] == new_samples[sample][key] for key in unchanged_fields)
    scope_file = HERE / 'rule_sync_20261025/nearest_point_review/before_approval/protected_scope.json'
    scope_before = json.loads(scope_file.read_text(encoding='utf-8'))
    scope_after = {}
    for relative in scope_before:
        root = HERE.parents[2] / relative
        paths = sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        checksum = hashlib.sha256()
        for path in paths:
            checksum.update(path.relative_to(root).as_posix().encode('utf-8'))
            checksum.update(hashlib.sha256(path.read_bytes()).digest())
        scope_after[relative] = dict(files=len(paths), sha256=checksum.hexdigest())
    assert scope_before == scope_after, 'A protected raw/publication/ML directory changed'
    result = dict(reclassified_curves=len(changed), other_curve_records_unchanged=len(before) - len(changed),
        previous_24_rising_points_unchanged=True, accepted_eta10_values_means_and_SD_unchanged=True,
        scan_limits_unchanged=True, approved_estimate_total_unchanged=70,
        protected_directory_hashes=scope_after,
        eta10_curves_sha256=hashlib.sha256((HERE / 'eta10_curves.csv').read_bytes()).hexdigest(),
        decisions_sha256=hashlib.sha256((REVIEW / 'approved_reassignments.csv').read_bytes()).hexdigest())
    (REVIEW / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify extraction against the fixed before snapshot')
    args = parser.parse_args()
    verify() if args.verify else prepare()
