"""Focused checks for pooling, cutoff semantics and explicit proxy labelling."""
import json
import statistics
import unittest

from build_overview import assemble, REPO, HERE
from lsv_plot_data import load_data, original_hd
from lsv_rules import automatic_selection
import numpy as np
import pandas as pd


class OverviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.counts = assemble()

    def test_cutoff_before_rounding(self):
        for value in [-20, -20.127298310399055, -19.99999, -13.658544048666954]:
            decision = automatic_selection(np.array([0., 1.]), np.array([value, value]))
            self.assertEqual(decision['status'] == 'nearest_point_estimate', -20 < value < -10)
        at_target = automatic_selection(np.array([0., 1.]), np.array([-10., -10.]))
        self.assertNotEqual(at_target['status'], 'nearest_point_estimate')

    def test_cofe_pools_individual_values_once(self):
        audit = pd.read_csv(HERE / 'CoFe_curve_audit.csv')
        values = audit.loc[audit.included_in_mean, 'eta10_mV'].tolist()
        row = next(row for row in self.rows if row['sample'] == 'Co-Fe')
        self.assertEqual(len(audit), 11)
        self.assertEqual(len(values), row['n_eta10'])
        self.assertAlmostEqual(statistics.mean(values), row['eta10_mean_mV'])
        self.assertAlmostEqual(statistics.stdev(values), row['eta10_SD_mV'])
        self.assertFalse(any(row['sample'] == 'reference_Co-Fe' for row in self.rows))
        self.assertEqual(row['n_HD'], 9)

    def test_proxy_not_mislabeled_as_crossing(self):
        row = next(row for row in self.rows if row['sample'] == 'Co-V11')
        self.assertEqual(row['n_eta10'], 0)
        self.assertIsNone(row['eta10_mean_mV'])
        self.assertEqual(row['n_plotted_potential'], 3)
        self.assertEqual(row['potential_metric'], 'estimate')
        self.assertAlmostEqual(row['plotted_potential_mean_mV'], 16.44449698130294)

    def test_no_fake_or_duplicate_coordinates(self):
        self.assertEqual(len(self.rows), len({row['sample'] for row in self.rows}))
        self.assertEqual(len(self.rows), 358)
        self.assertTrue(all(row['plot_location'] == 'main' for row in self.rows))
        self.assertEqual(self.counts, {'both': 97, 'potential_only': 255, 'HD_only': 6})

    def test_all_audited_recipes_pool_individual_values(self):
        evidence = json.loads((HERE / 'recipe_evidence.json').read_text(encoding='utf-8'))
        curves, _ = load_data()
        measurements = pd.read_csv(HERE.parent / 'HD_repeat_analysis/tables/measurement_sources.csv')
        self.assertTrue(measurements.measurement_id.is_unique)
        self.assertEqual(len(measurements), 806)
        for group in evidence['groups']:
            name = group['sample']
            self.assertEqual(group['initial']['nonzero_recipe'], group['BO_reference']['nonzero_recipe'])
            source = curves.loc[curves['sample'].eq(name)]
            self.assertEqual(len(source), source.sha256.nunique())
            accepted = source.loc[source.status.eq('eta10_crossing'), 'eta10_mV']
            row = next(r for r in self.rows if r['sample'] == name)
            self.assertAlmostEqual(row['eta10_mean_mV'], accepted.mean())
            self.assertAlmostEqual(row['eta10_SD_mV'], accepted.std(ddof=1))
            hd = measurements.loc[measurements['sample'].eq(name), 'S_HD']
            self.assertAlmostEqual(row['HD_mean'], hd.mean())
            self.assertAlmostEqual(row['HD_SD'], hd.std(ddof=1))
            self.assertEqual(row['n_HD'], 9 if name == 'Co-Fe' else 8)
        self.assertFalse(any(r['sample'].startswith('reference_') for r in self.rows))
        raw_hd = pd.read_csv(REPO / '0-Figures/Figure1/revision_figure/figure1b_inputs/original_HD.csv')
        self.assertEqual(int(original_hd().original_HD_count.sum()), int(raw_hd.original_HD_count.sum()))

    def test_figure1_and_overview_use_identical_lsv_values(self):
        _, stats = load_data()
        stats = stats.set_index('sample')
        for row in self.rows:
            self.assertAlmostEqual(row['plotted_potential_mean_mV'], stats.loc[row['sample'], 'potential_mean_mV'])
            if row['plotted_potential_SD_mV'] is not None:
                self.assertAlmostEqual(row['plotted_potential_SD_mV'], stats.loc[row['sample'], 'potential_SD_mV'])

    def test_all_estimates_are_explicit(self):
        selected = [r for r in self.rows if r['potential_metric'] == 'estimate']
        self.assertEqual({r['sample'] for r in selected}, {'Co-V11', 'Eu-Pd', 'W-Pd'})
        self.assertTrue(all(r['eta10_mean_mV'] is None and r['eta10_SD_mV'] is None for r in selected))
        self.assertTrue(all(r['plotted_potential_SD_mV'] is not None for r in selected))


if __name__ == '__main__':
    unittest.main()
