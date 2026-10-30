import unittest

import numpy as np

from compare_automatic_rising import automatic_selection, summarize


class AutomaticRisingTests(unittest.TestCase):
    def curve(self, potentials, currents):
        eta = np.arange(potentials[0], potentials[-1] + 1, dtype=float)
        return eta, np.interp(eta, potentials, currents)

    def test_terminal_gate_overrides_crossing_and_rise(self):
        eta, j = self.curve([0, 100, 200, 300, 400], [0, -30, -15, -40, -5])
        out = automatic_selection(eta, j)
        self.assertEqual(out['status'], 'scan_limit')
        self.assertEqual(out['potential_mV'], 400)
        self.assertEqual(out['candidates'], [])

    def test_scan_limit_is_maximum_not_final_potential(self):
        out = automatic_selection([0, 300, 100], [0, -20, -5])
        self.assertEqual((out['status'], out['potential_mV'], out['j_mA_cm2']), ('scan_limit', 300, -20))

    def test_normal_monotonic_crossing(self):
        out = automatic_selection([0, 100, 200], [0, -20, -40])
        self.assertEqual((out['status'], out['potential_mV']), ('eta10_crossing', 50))

    def test_final_exact_target_is_crossing(self):
        out = automatic_selection([0, 100], [0, -10])
        self.assertEqual((out['status'], out['potential_mV']), ('eta10_crossing', 100))

    def test_later_rise_overrides_early_crossing_without_approval(self):
        eta, j = self.curve([0, 100, 200, 400], [0, -30, -15, -50])
        out = automatic_selection(eta, j)
        self.assertEqual(out['status'], 'rising_point')
        self.assertGreater(out['potential_mV'], out['largest_crossing_mV'])
        self.assertTrue(-20 < out['j_mA_cm2'] < -10)
        self.assertEqual(out['potential_mV'], eta[out['before']])
        self.assertEqual(out['j_mA_cm2'], j[out['before']])

    def test_multiple_eligible_rises_choose_largest_potential(self):
        eta, j = self.curve([0, 50, 100, 200, 300, 500], [-12, -30, -14, -35, -18, -60])
        out = automatic_selection(eta, j)
        eligible = [c for c in out['candidates'] if c['eligible']]
        self.assertEqual(len(eligible), 2)
        self.assertEqual(out['potential_mV'], max(c['potential_mV'] for c in eligible))
        self.assertGreater(out['potential_mV'], 290)

    def test_later_rise_outside_current_window_is_not_selected(self):
        eta, j = self.curve([0, 50, 100, 200, 300, 500], [-12, -30, -14, -35, -25, -60])
        out = automatic_selection(eta, j)
        self.assertEqual(out['status'], 'rising_point')
        self.assertLess(out['potential_mV'], 110)
        self.assertTrue(any(not c['within_current_window'] for c in out['candidates']))

    def test_rise_before_last_crossing_cannot_override_it(self):
        eta, j = self.curve([0, 100, 200, 300, 400, 500], [0, -30, -12, -30, -5, -30])
        out = automatic_selection(eta, j)
        self.assertEqual(out['status'], 'eta10_crossing')
        self.assertEqual(out['potential_mV'], out['largest_crossing_mV'])
        self.assertTrue(any(c['within_current_window'] and not c['after_last_crossing'] for c in out['candidates']))

    def test_no_crossing_or_rise_falls_back_to_nearest(self):
        out = automatic_selection([0, 100, 200], [-12, -30, -40])
        self.assertEqual((out['status'], out['potential_mV']), ('nearest_point_estimate', 0))

    def test_nearest_cutoff_is_strict_before_rounding(self):
        for j in (-20, -20.000001, -30):
            with self.subTest(j=j):
                self.assertEqual(automatic_selection([0, 100], [j, -40])['status'], 'above_target_no_crossing')
        self.assertEqual(automatic_selection([0, 100], [-19.999999, -40])['status'], 'nearest_point_estimate')

    def test_reversal_retains_crossing_and_flag(self):
        out = automatic_selection([0, 300, 100, 200], [0, -20, 0, -20])
        self.assertEqual((out['status'], out['potential_mV']), ('eta10_crossing', 200))
        self.assertIn('reversal', out['detector_diagnostic'])

    def test_summary_never_pools_estimates_or_bounds_with_eta10(self):
        rows = [dict(sample='test', status='eta10_crossing', eta10_mV=100),
                dict(sample='test', status='eta10_crossing', eta10_mV=120),
                dict(sample='test', status='rising_point', rising_point_potential_mV=200),
                dict(sample='test', status='nearest_point_estimate', nearest_point_potential_mV=10),
                dict(sample='test', status='scan_limit', scan_limit_mV=550)]
        out = summarize(rows)['test']
        self.assertEqual((out['n_eta10'], out['eta10_mean_mV']), (2, 110))
        self.assertAlmostEqual(out['eta10_SD_mV'], np.sqrt(200))
        self.assertEqual(out['rising_point_mean_mV'], 200)
        self.assertIsNone(out['rising_point_SD_mV'])


class NoiseExceptionTests(unittest.TestCase):
    def context(self):
        from extract_eta10 import RAW, load_curve, load_noise_exceptions, read_csv
        exceptions = load_noise_exceptions()
        self.assertEqual(len(exceptions), 1)
        exception = next(iter(exceptions.values()))
        records = {r['curve_id']: r for r in read_csv(RAW / 'manifest.csv')}
        record = records[exception['curve_id']]
        self.assertEqual(record['sha256'], exception['sha256'])
        self.assertEqual(record['file'], 'repeat/Co-Ce/repeat4_Co-Ce.txt')
        _, eta, j, _, _, _ = load_curve(record)
        return exception, eta, j

    def test_coce_keeps_crossing_and_preserves_rejected_automatic_point(self):
        from lsv_rules import apply_noise_exception
        exception, eta, j = self.context()
        auto = automatic_selection(eta, j)
        retained = apply_noise_exception(auto, eta, j, exception)
        self.assertEqual(auto['status'], 'rising_point')
        self.assertAlmostEqual(auto['potential_mV'], 273.7473375320435)
        self.assertEqual(retained['status'], 'eta10_crossing')
        self.assertAlmostEqual(retained['potential_mV'], 135.7661872215634)
        self.assertEqual(retained['j_mA_cm2'], -10)

    def test_stale_noise_exception_requires_review(self):
        from lsv_rules import apply_noise_exception
        exception, eta, j = self.context()
        stale = dict(exception, rejected_automatic_potential_mV=100)
        with self.assertRaises(AssertionError):
            apply_noise_exception(automatic_selection(eta, j), eta, j, stale)

    def test_exception_cannot_bypass_terminal_gate(self):
        from lsv_rules import apply_noise_exception
        exception, _, _ = self.context()
        eta, j = [0, 100, 200], [-12, -30, -5]
        with self.assertRaises(AssertionError):
            apply_noise_exception(automatic_selection(eta, j), eta, j, exception)

    def test_shared_rule_matches_frozen_automatic_proposal(self):
        from extract_eta10 import HERE, RAW, load_curve, read_csv
        records = {r['curve_id']: r for r in read_csv(RAW / 'manifest.csv')}
        proposal = read_csv(HERE / 'automatic_rising_review_20261025/proposed_curves.csv')
        self.assertEqual(len(proposal), 1583)
        for row in proposal:
            _, eta, j, _, _, _ = load_curve(records[row['curve_id']])
            decision = automatic_selection(eta, j)
            self.assertEqual(decision['status'], row['status'])
            for key, field in (('potential_mV', 'selected_potential_mV'), ('j_mA_cm2', 'selected_j_mA_cm2')):
                if row[field] == '':
                    self.assertIsNone(decision[key])
                else:
                    self.assertAlmostEqual(decision[key], float(row[field]), places=6)


if __name__ == '__main__':
    unittest.main()
