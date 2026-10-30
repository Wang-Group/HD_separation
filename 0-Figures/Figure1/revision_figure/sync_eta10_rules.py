"""Carry forward reviewed raw-point decisions and audit the unified extraction."""
import argparse
import hashlib
import json
from collections import Counter

from extract_eta10 import HERE, RAW, load_approvals, load_curve, read_csv, write_csv

REPO = HERE.parents[2]
REVIEW = HERE / 'rule_sync_20261025'
INITIAL_DECISIONS = HERE.parent / 'data/figure1_this_work_eta10_censoring.csv'
COV_DECISIONS = REPO / 'others_revision/reproducibility_overview/CoV11_closest_point_audit.csv'
NEAREST_APPROVAL = REVIEW / 'nearest_point_review/approval.json'
REVIEWED_TERMINAL_SAMPLES = (
    'Cd-Al', 'Cd-Ce', 'Cd-Eu', 'Ce-Al', 'Ce-Eu', 'Eu-In', 'Eu-Pb', 'Ga-Mo',
    'Ga-Pb', 'In-Eu', 'In-La', 'In-Mo', 'La-Pb', 'Sc-Zn', 'Y-Ba', 'Y-Pb',
    'Zn-Ga', 'Zn-Zn',
)


def prepare_approvals():
    manifest = read_csv(RAW / 'manifest.csv')
    by_source = {r['source_relative_path']: r for r in manifest}
    assert len(by_source) == len(manifest)
    records = {}

    def add(source, potential, current, kind, evidence, basis, raw_line=None):
        record = by_source[source]
        points, eta, j, _, _, _ = load_curve(record)
        matches = [k for k in range(len(points)) if
                   abs(eta[k] - potential) < 1e-6 and abs(j[k] - current) < 1e-6]
        if raw_line is not None:
            matches = [k for k in matches if points[k][0] == int(raw_line)]
        assert len(matches) == 1, source
        k = matches[0]
        assert -20 < j[k] < -10, source
        if kind == 'nearest_point_estimate':
            assert abs(abs(j[k] + 10) - min(abs(v + 10) for v in j)) < 1e-9
        if source.startswith('LSV/'):
            original = REPO / 'hd_separation_ml/data_extraction/all_data' / source
            assert hashlib.sha256(original.read_bytes()).hexdigest() == record['sha256']
        row = dict(sample=record['sample_group'], source_relative_path=source,
                   curve_id=record['curve_id'], sha256=record['sha256'],
                   raw_line=points[k][0], expected_potential_mV=eta[k],
                   expected_j_mA_cm2=j[k], estimate_type=kind, approval_basis=basis,
                   evidence_file=evidence.relative_to(REPO).as_posix(),
                   evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
        previous = records.get(record['curve_id'])
        if previous:
            assert previous['raw_line'] == row['raw_line']
            assert previous['estimate_type'] == row['estimate_type']
        else:
            records[record['curve_id']] = row

    initial = read_csv(INITIAL_DECISIONS)
    for row in initial:
        status = row['eta10_status']
        if status == 'left_censored_near_target_estimate':
            potential = float(row['nearest_to_target_eta_estimate_mV'])
            current = float(row['nearest_j_to_target_mA_cm2'])
            kind = 'nearest_point_estimate'
        elif status == 'left_censored_turning_point_estimate':
            potential = float(row['turning_point_eta_estimate_mV'])
            current = float(row['turning_point_j_mA_cm2'])
            kind = 'rising_point'
        else:
            continue
        add('LSV/' + row['sample'] + '.txt', potential, current, kind,
            INITIAL_DECISIONS, 'initial_submission_Figure1_review_20260618')
    assert len(records) == 13
    for row in read_csv(COV_DECISIONS):
        if row['included_in_proxy_mean'] != 'True':
            continue
        assert by_source[row['source_relative_path']]['sha256'] == row['sha256']
        add(row['source_relative_path'], float(row['closest_potential_mV']),
            float(row['closest_j_mA_cm2']), 'nearest_point_estimate', COV_DECISIONS,
            'CoV11_explicit_closest_point_review_20261025')
    # The original Co-V11 is present in both evidence files, but only once here.
    assert len(records) == 15
    approval = json.loads(NEAREST_APPROVAL.read_text(encoding='utf-8'))
    reviewed_file = NEAREST_APPROVAL.parent / approval['reviewed_points_file']
    assert reviewed_file.resolve().parent == NEAREST_APPROVAL.parent.resolve()
    assert hashlib.sha256(reviewed_file.read_bytes()).hexdigest() == approval['reviewed_points_sha256']
    assert approval['decision'] == 'nearest_point_estimate'
    assert approval['eta10_field_must_remain_blank'] and not approval['pool_with_eta10']
    reviewed = read_csv(reviewed_file)
    assert len(reviewed) == len({r['curve_id'] for r in reviewed}) == approval['curve_count'] == 35
    assert not (set(records) & {r['curve_id'] for r in reviewed})
    for row in reviewed:
        record = by_source[row['source_relative_path']]
        assert (record['curve_id'], record['sha256']) == (row['curve_id'], row['sha256'])
        add(row['source_relative_path'], float(row['candidate_potential_mV']),
            float(row['candidate_j_mA_cm2']), 'nearest_point_estimate', NEAREST_APPROVAL,
            'explicit_35_curve_nearest_point_review_20261025', raw_line=row['raw_line'])
    assert len(records) == 50
    reassignment_file = REVIEW / 'rising_reclassification/approved_reassignments.csv'
    reassignments = read_csv(reassignment_file)
    assert len(reassignments) == len({r['curve_id'] for r in reassignments}) == 39
    for decision in reassignments:
        previous = records[decision['curve_id']]
        assert previous['estimate_type'] == 'nearest_point_estimate'
        assert previous['sha256'] == decision['sha256']
        assert int(previous['raw_line']) == int(decision['previous_raw_line'])
        assert abs(float(previous['expected_potential_mV']) - float(decision['previous_potential_mV'])) < 1e-6
        assert abs(float(previous['expected_j_mA_cm2']) - float(decision['previous_j_mA_cm2'])) < 1e-6
        previous.update(raw_line=int(decision['raw_line']),
            expected_potential_mV=float(decision['expected_potential_mV']),
            expected_j_mA_cm2=float(decision['expected_j_mA_cm2']), estimate_type='rising_point',
            approval_basis=decision['approval_basis'],
            evidence_file=reassignment_file.relative_to(REPO).as_posix(),
            evidence_sha256=hashlib.sha256(reassignment_file.read_bytes()).hexdigest())
    existing = {r['curve_id'] for r in read_csv(HERE / 'approved_rising_points.csv')}
    assert not (existing & set(records))
    write_csv(HERE / 'approved_additional_estimates.csv',
              sorted(records.values(), key=lambda r: r['source_relative_path']))
    print(json.dumps(dict(additional_approved_points=len(records),
                         by_type=dict(Counter(r['estimate_type'] for r in records.values()))), indent=2))


def verify_sync():
    old = {r['curve_id']: r for r in read_csv(REVIEW / 'before/eta10_curves.csv')}
    rows = read_csv(HERE / 'eta10_curves.csv')
    assert set(old) == {r['curve_id'] for r in rows}
    changes = []
    for row in rows:
        previous = old[row['curve_id']]
        assert row['sha256'] == previous['sha256']
        for key in ('eta10_mV', 'scan_limit_mV'):
            assert row[key] == previous[key], (row['file'], key)
        if previous['status'] == 'rising_point':
            assert row['status'] == 'rising_point'
            assert row['rising_point_potential_mV'] == previous['rising_point_potential_mV']
            assert row['rising_point_j_mA_cm2'] == previous['rising_point_j_mA_cm2']
        if row['approved_estimate_potential_mV']:
            assert not row['eta10_mV'] and not row['scan_limit_mV']
            assert -20 < float(row['approved_estimate_j_mA_cm2']) < -10
        if row['status'] != previous['status']:
            assert previous['status'] == 'above_target_no_crossing'
            changes.append(dict(sample=row['sample'], file=row['file'],
                old_status=previous['status'], new_status=row['status'],
                estimate_potential_mV=row['approved_estimate_potential_mV'],
                estimate_j_mA_cm2=row['approved_estimate_j_mA_cm2'],
                approval_basis=row['approval_basis'], source_relative_path=row['source_relative_path'],
                curve_id=row['curve_id'], sha256=row['sha256']))
    approved = load_approvals()
    expected_changes = {key for key in approved if old[key]['status'] == 'above_target_no_crossing'}
    assert {r['curve_id'] for r in changes} == expected_changes
    assert len(changes) == 50
    write_csv(REVIEW / 'classification_changes.csv', changes)
    current_by_source = {r['source_relative_path']: r for r in rows}
    terminal = []
    for sample in REVIEWED_TERMINAL_SAMPLES:
        row = current_by_source['LSV/' + sample + '.txt']
        assert row['status'] == 'scan_limit' and int(row['crossing_count']) > 0
        assert float(row['final_j_mA_cm2']) > -10
        assert row['scan_limit_mV'] == row['max_eta_mV']
        terminal.append(dict(sample=sample, scan_limit_mV=row['scan_limit_mV'],
            final_j_mA_cm2=row['final_j_mA_cm2'], largest_crossing_mV=row['largest_crossing_mV'],
            source_relative_path=row['source_relative_path'], curve_id=row['curve_id']))
    write_csv(REVIEW / 'reviewed_18_scan_limits.csv', terminal)
    for sample in ('Sc-Pb', 'Cd-Y', 'W-Pb', 'Ce-Pb'):
        assert current_by_source['LSV/' + sample + '.txt']['status'] == 'scan_limit'
    cov = [r for r in rows if r['sample'] == 'Co-V11']
    assert Counter(r['status'] for r in cov) == {
        'nearest_point_estimate': 1, 'rising_point': 2, 'above_target_no_crossing': 1}
    old_samples = {r['sample']: r for r in read_csv(REVIEW / 'before/eta10_samples.csv')}
    samples = read_csv(HERE / 'eta10_samples.csv')
    assert set(old_samples) == {r['sample'] for r in samples}
    for row in samples:
        for key in ('raw_curves', 'n_eta10', 'eta10_mean_mV', 'eta10_SD_mV', 'n_scan_limit'):
            assert row[key] == old_samples[row['sample']][key], (row['sample'], key)
    count_rows = []
    for origin in ('original', 'additional', 'all'):
        part = rows if origin == 'all' else [r for r in rows if r['origin'] == origin]
        counts = Counter(r['status'] for r in part)
        count_rows.append(dict(origin=origin, raw_curves=len(part),
            accepted_eta10=counts['eta10_crossing'], approved_rising=counts['rising_point'],
            approved_nearest=counts['nearest_point_estimate'],
            approved_estimates=counts['rising_point'] + counts['nearest_point_estimate'],
            scan_limits=counts['scan_limit'], no_accepted_crossing_or_estimate=counts['above_target_no_crossing'],
            needs_review=counts['needs_review']))
    write_csv(REVIEW / 'counts.csv', count_rows)
    unapproved = [dict(sample=r['sample'], file=r['file'], origin=r['origin'],
                      nearest_potential_mV=r['nearest_candidate_potential_mV'],
                      nearest_j_mA_cm2=r['nearest_candidate_j_mA_cm2'],
                      status=r['status'], adopted=False,
                      reason='Numerical window alone is not approval; retain existing non-adoption',
                      source_relative_path=r['source_relative_path'], curve_id=r['curve_id'], sha256=r['sha256'])
                  for r in rows if r['status'] == 'above_target_no_crossing'
                  and r['nearest_candidate_within_estimate_window'] == 'True']
    # Preserve the original 35-curve review input as evidence of what was shown.
    fields = list(read_csv(REVIEW / 'unapproved_nearest_candidates.csv')[0])
    write_csv(REVIEW / 'remaining_unapproved_nearest_candidates.csv', unapproved, fieldnames=fields)
    summary = dict(changed_classifications=len(changes), all_accepted_eta10_unchanged=True,
        all_scan_limits_unchanged=True, existing_20_rising_points_unchanged=True,
        accepted_eta10_sample_means_and_SD_unchanged=True,
        reviewed_18_retained_as_scan_limits=True, known_four_artifacts_not_accepted=True,
        curve_ids_and_raw_hashes_unchanged=True, CoV11_counts=dict(Counter(r['status'] for r in cov)),
        counts=count_rows, unapproved_nearest_candidates_within_window=len(unapproved),
        publication_figures_refreshed=False, ML_rerun=False)
    (REVIEW / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-approvals', action='store_true')
    args = parser.parse_args()
    if args.prepare_approvals:
        prepare_approvals()
    else:
        verify_sync()
