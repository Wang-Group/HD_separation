import unittest
from extract_eta10 import approved_selection, load_approvals, load_curve, RAW, read_csv, select


class ExtractionTests(unittest.TestCase):
    def test_linear_interpolation(self):
        self.assertEqual(select([0, 100], [0, -20])[:3], ('eta10_crossing', 50, -10))

    def test_largest_crossing(self):
        self.assertEqual(select([0, 100, 200, 300], [0, -20, 0, -20])[1], 250)

    def test_terminal_gate_overrides_earlier_crossing(self):
        self.assertEqual(select([0, 100, 200], [0, -20, -5])[:2], ('scan_limit', 200))

    def test_never_reaches(self):
        self.assertEqual(select([0, 100], [-1, -9])[:2], ('scan_limit', 100))

    def test_all_above(self):
        self.assertEqual(select([0, 100], [-12, -20])[:2], ('above_target_no_crossing', None))

    def test_start_above_then_crosses(self):
        self.assertEqual(select([0, 100, 200], [-20, 0, -20])[1], 150)

    def test_plateau(self):
        result = select([0, 100, 200, 300], [0, -10, -10, -20])
        self.assertEqual(result[1], 200)
        self.assertEqual(len(result[-1]), 1)

    def test_reversal_uses_largest_eta_not_last_index(self):
        self.assertEqual(select([0, 300, 100, 200], [0, -20, 0, -20])[1], 200)

    def test_final_exact_target(self):
        self.assertEqual(select([0, 100], [0, -10])[:2], ('eta10_crossing', 100))

    def test_boundary_is_scan_maximum(self):
        self.assertEqual(select([0, 300, 100], [0, -8, -5])[:2], ('scan_limit', 300))


class EstimateTests(unittest.TestCase):
    def decision(self, kind, eta, j, k):
        return dict(estimate_type=kind, raw_line=k + 1,
                    expected_potential_mV=eta[k], expected_j_mA_cm2=j[k])

    def apply(self, eta, j, kind='nearest_point_estimate', k=0):
        points = [(i + 1,) for i in range(len(j))]
        return approved_selection(eta, j, points, self.decision(kind, eta, j, k))

    def test_nearest_does_not_require_whole_curve_below_twenty(self):
        result = self.apply([5, 50, 100], [-12, -30, -50])
        self.assertEqual(result[:3], ('nearest_point_estimate', 5, -12))

    def test_no_approval_means_no_estimate(self):
        self.assertEqual(select([5, 100], [-12, -30])[0], 'above_target_no_crossing')

    def test_strict_cutoff_before_rounding(self):
        for current in (-20, -20.127, -20.0000001):
            with self.subTest(current=current), self.assertRaises(AssertionError):
                self.apply([5, 100], [current, -40])
        self.assertEqual(self.apply([5, 100], [-19.999999, -40])[0], 'nearest_point_estimate')

    def test_actual_target_is_not_estimate(self):
        with self.assertRaises(AssertionError):
            self.apply([5, 100], [-10, -40])

    def test_nearest_must_be_global_closest(self):
        with self.assertRaises(AssertionError):
            self.apply([5, 50, 100], [-12, -15, -40], k=1)

    def test_rising_point_need_not_be_nearest(self):
        result = self.apply([5, 50, 100], [-12, -15, -40], kind='rising_point', k=1)
        self.assertEqual(result[:3], ('rising_point', 50, -15))

    def test_estimate_cannot_override_scan_limit(self):
        with self.assertRaises(AssertionError):
            self.apply([0, 50, 100], [-12, -30, -5])

    def test_nearest_cannot_replace_crossing(self):
        with self.assertRaises(AssertionError):
            self.apply([0, 50, 100], [-12, -5, -30])

    def test_all_registered_points_recalculate_and_pass(self):
        manifest = {r['curve_id']: r for r in read_csv(RAW / 'manifest.csv')}
        approved = load_approvals()
        self.assertEqual(len(approved), 70)
        for curve_id, decision in approved.items():
            with self.subTest(curve_id=curve_id):
                record = manifest[curve_id]
                self.assertEqual(record['sha256'], decision['sha256'])
                points, eta, j, _, _, _ = load_curve(record)
                result = approved_selection(eta, j, points, decision)
                self.assertEqual(result[0], decision['estimate_type'])

    def test_duplicate_readings_keep_the_approved_raw_line(self):
        result = self.apply([-6, -6, 100], [-12, -12, -40], k=1)
        self.assertEqual(result[:5], ('nearest_point_estimate', -6, -12, 1, 1))

    def test_all_35_reviewed_points_are_registered_exactly(self):
        from extract_eta10 import HERE
        reviewed = read_csv(HERE / 'rule_sync_20261025/nearest_point_review/plotted_candidates.csv')
        previous = {r['curve_id']: r for r in read_csv(
            HERE / 'rule_sync_20261025/rising_reclassification/before/approved_additional_estimates.csv')}
        updates = {r['curve_id']: r for r in read_csv(
            HERE / 'rule_sync_20261025/rising_reclassification/approved_reassignments.csv')}
        approved = load_approvals()
        self.assertEqual(len(reviewed), 35)
        for row in reviewed:
            decision = previous[row['curve_id']]
            self.assertEqual(decision['estimate_type'], 'nearest_point_estimate')
            self.assertEqual(decision['raw_line'], row['raw_line'])
            self.assertEqual(decision['sha256'], row['sha256'])
            self.assertEqual(float(decision['expected_potential_mV']), float(row['candidate_potential_mV']))
            self.assertEqual(float(decision['expected_j_mA_cm2']), float(row['candidate_j_mA_cm2']))
            current = approved[row['curve_id']]
            update = updates.get(row['curve_id'])
            if update:
                self.assertEqual(current['estimate_type'], 'rising_point')
                self.assertEqual(current['raw_line'], update['raw_line'])
            else:
                self.assertEqual(current, decision)


class LatestRisingTests(unittest.TestCase):
    def test_largest_potential_not_closest_current(self):
        from reclassify_nearest_as_rising import choose_latest
        candidates = [dict(potential_mV=20, j_mA_cm2=-10.01),
                      dict(potential_mV=120, j_mA_cm2=-18)]
        self.assertEqual(choose_latest(candidates), candidates[1])

    def test_later_point_outside_window_is_not_eligible(self):
        from reclassify_nearest_as_rising import choose_latest
        candidates = [dict(potential_mV=20, j_mA_cm2=-12),
                      dict(potential_mV=120, j_mA_cm2=-20),
                      dict(potential_mV=220, j_mA_cm2=-10)]
        self.assertEqual(choose_latest(candidates), candidates[0])

    def test_monotonic_curve_has_no_rising_candidate(self):
        import numpy as np
        from reclassify_nearest_as_rising import rising_candidates, choose_latest
        eta = np.arange(0, 301, 2)
        self.assertEqual(rising_candidates(eta, -12 - .1 * eta), [])
        self.assertIsNone(choose_latest([]))

    def test_all_reassignments_use_latest_eligible_point(self):
        from extract_eta10 import HERE
        from reclassify_nearest_as_rising import rising_candidates, choose_latest
        records = {r['curve_id']: r for r in read_csv(RAW / 'manifest.csv')}
        updates = read_csv(HERE / 'rule_sync_20261025/rising_reclassification/approved_reassignments.csv')
        self.assertEqual(len(updates), 39)
        for row in updates:
            points, eta, j, _, _, _ = load_curve(records[row['curve_id']])
            latest = choose_latest(rising_candidates(eta, j))
            self.assertEqual(float(row['expected_potential_mV']), latest['potential_mV'])
            self.assertEqual(float(row['expected_j_mA_cm2']), latest['j_mA_cm2'])
            self.assertEqual(int(row['raw_line']), points[latest['index']][0])


class RecordSeparationTests(unittest.TestCase):
    def test_adopted_metrics_use_exclusive_columns(self):
        from extract_eta10 import HERE
        groups = {
            'eta10_crossing': {'eta10_mV'},
            'rising_point': {'rising_point_potential_mV', 'rising_point_j_mA_cm2'},
            'nearest_point_estimate': {'nearest_point_potential_mV', 'nearest_point_j_mA_cm2'},
            'scan_limit': {'scan_limit_mV'},
            'above_target_no_crossing': set(),
        }
        fields = set().union(*groups.values())
        for row in read_csv(HERE / 'eta10_curves.csv'):
            with self.subTest(curve_id=row['curve_id']):
                self.assertEqual({key for key in fields if row[key] != ''}, groups[row['status']])
                is_estimate = row['status'] in ('rising_point', 'nearest_point_estimate')
                for key in ('approved_estimate_potential_mV', 'approved_estimate_j_mA_cm2', 'approval_basis'):
                    self.assertEqual(row[key] != '', is_estimate)
                self.assertEqual(row['scan_limit_reason'] != '', row['status'] == 'scan_limit')

    def test_empty_candidate_csv_has_header_and_no_rows(self):
        import tempfile
        from pathlib import Path
        from extract_eta10 import write_csv
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.csv'
            write_csv(path, [], fieldnames=['curve_id', 'status'])
            self.assertEqual(read_csv(path), [])
            self.assertEqual(path.read_text(encoding='utf-8-sig').strip(), 'curve_id,status')


if __name__ == '__main__':
    unittest.main()
