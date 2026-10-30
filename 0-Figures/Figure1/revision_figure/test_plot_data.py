"""Cross-output consistency and scientific category regression tests."""
import json
import unittest
import numpy as np
import pandas as pd
from lsv_plot_data import HERE, REPO, ALIASES, load_data, original_hd, mean_sd


class PlotDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curves, stats = load_data()
        cls.stats = stats.set_index('sample')
        cls.figure = pd.read_csv(HERE/'figure1b_outputs/plotted_samples.csv').set_index('sample')
        cls.overview = pd.read_csv(REPO/'others_revision/reproducibility_overview/overview_points.csv').set_index('sample')

    def test_curve_counts_and_means(self):
        self.assertEqual(len(self.curves), 1583)
        self.assertEqual(len(self.stats), 672)
        self.assertEqual(int(self.stats.n_eta10.sum()), 1105)
        self.assertEqual(int(self.stats.n_eta10.ge(2).sum()), 349)
        self.assertEqual(int(self.stats.n_estimates.sum()), 71)
        self.assertEqual(int(self.stats.n_scan_limit.sum()), 382)
        for sample, group in self.curves.groupby('sample'):
            eta = group.loc[group.status.eq('eta10_crossing'), 'eta10_mV']
            np.testing.assert_allclose(mean_sd(eta), self.stats.loc[sample, ['eta10_mean_mV','eta10_SD_mV']].astype(float), equal_nan=True)

    def test_same_recipe_pool_and_original_hd(self):
        row = self.stats.loc['Co-Fe']
        self.assertEqual(row.raw_curves, 11)
        self.assertEqual(row.n_eta10, 10)
        self.assertNotIn('reference_Co-Fe', self.stats.index)
        hd = original_hd().set_index('sample')
        raw = pd.read_csv(HERE/'figure1b_inputs/original_HD.csv').set_index('sample')
        for alias, sample in ALIASES.items():
            source = raw.loc[[sample, alias]]
            self.assertAlmostEqual(hd.loc[sample,'original_HD_coefficient'], np.average(source.original_HD_coefficient,weights=source.original_HD_count))
            self.assertNotIn(alias, self.stats.index)
        unchanged = raw.index.difference(list(ALIASES) + list(ALIASES.values()))
        np.testing.assert_array_equal(hd.loc[unchanged,'original_HD_coefficient'],raw.loc[unchanged,'original_HD_coefficient'])

    def test_estimates_and_limits_not_eta10(self):
        selected = self.stats.loc[self.stats.potential_metric.ne('eta10')]
        self.assertTrue(selected.eta10_mean_mV.isna().all())
        self.assertTrue(selected.eta10_SD_mV.isna().all())
        self.assertEqual(self.stats.loc['Co-V11','n_estimates'],3)
        self.assertAlmostEqual(self.stats.loc['Co-V11','estimate_mean_mV'],16.44449698130294)
        self.assertEqual(self.stats.loc['Ga-Cd','potential_metric'],'estimate')
        self.assertTrue(np.isnan(self.stats.loc['Ga-Cd','eta10_mean_mV']))
        self.assertEqual(self.stats.loc['Ga-Cd','n_scan_limit'],4)

    def test_cross_output_coordinates_and_uncapped_sd(self):
        for table, mean, sd in [(self.figure,'x','sd'),(self.overview,'plotted_potential_mean_mV','plotted_potential_SD_mV')]:
            source = self.stats.loc[table.index]
            np.testing.assert_allclose(table[mean],source.potential_mean_mV,equal_nan=True)
            np.testing.assert_allclose(table[sd],source.potential_SD_mV,equal_nan=True)
        self.assertTrue(self.stats.eta10_mean_mV.gt(550).any())
        self.assertTrue(self.figure.display_x.between(10,550).all())
        self.assertTrue(self.figure.loc[self.figure.metric.eq('scan_limit'),'sd'].isna().all())

    def test_frontier_exclusions_and_noise_exception(self):
        self.assertTrue(self.figure.loc[self.figure.on_frontier,'metric'].eq('eta10').all())
        self.assertFalse(self.figure.loc[self.figure.on_frontier,'is_pure_metal'].any())
        coce = self.curves.loc[self.curves.file.eq('repeat/Co-Ce/repeat4_Co-Ce.txt')].iloc[0]
        self.assertEqual(coce.status,'eta10_crossing')
        self.assertAlmostEqual(coce.eta10_mV,135.7661872215634)
        self.assertAlmostEqual(self.stats.loc['Co-Ce','eta10_mean_mV'],123.54347822616637)

    def test_repeatability_distribution_same_samples(self):
        sd = pd.read_csv(HERE/'repeatability_review/sample_SD_distribution.csv').set_index('sample')
        source = self.stats.loc[self.stats.n_eta10.ge(2)]
        self.assertEqual(set(sd.index),set(source.index))
        np.testing.assert_allclose(sd.eta10_SD_mV,source.loc[sd.index,'eta10_SD_mV'])
        info = json.loads((HERE/'figure1b_outputs/provenance.json').read_text())
        self.assertEqual(info['figure_size_inches'],[16,8])
        self.assertEqual(info['display_window'],[10,550])


if __name__ == '__main__':
    unittest.main()
