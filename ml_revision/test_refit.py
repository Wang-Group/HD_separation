"""Target, transform, split and selection regression tests (no full ML fits)."""
import json
import unittest
import numpy as np
import pandas as pd
from cohorts import prepare, eta_targets, memberships
from descriptors import DescriptorTransform, COVARIATES
from model_core import estimator
from selector import correlation_groups, choose_candidates, fit_selector, resample_splits
from workflow.data import ROOT, load_data, FEATURES


class RefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text())
        cls.jobs = prepare(cls.config)
        cls.data, cls.audit = load_data()

    def test_cohorts(self):
        self.assertEqual(len(self.jobs), 285)
        self.assertEqual(self.data.cohort.value_counts().to_dict(), {'initial':560, 'BO':111})
        for j in self.jobs:
            self.assertTrue(j['test'].cohort.eq('initial').all())
            if j['arm'] != 'with_bo':
                self.assertTrue(j['train'].cohort.eq('initial').all())

    def test_removed_descriptors(self):
        self.assertEqual(len(FEATURES), 30)
        self.assertFalse(any(x in f for x in ('MD_bond', 'Frequency_D', 'ZPE') for f in FEATURES))

    def test_transform_size_shift(self):
        j = next(j for j in self.jobs if j['workflow']=='ec_shifted_measured')
        d = j['train']
        base = DescriptorTransform('icp').fit(d).transform(d)
        shifted = DescriptorTransform('shifted_measured').fit(d).transform(d)
        self.assertEqual(shifted.shape[1], 33)
        for b, sign in [('work_function',1),('pzc',-1),('E_std',-1)]:
            for c in [b+'_element1',b+'_element2','icp_weighted_'+b]:
                np.testing.assert_allclose(shifted[c]-base[c], sign*(d.E_IR_corrected_final+.098))
        np.testing.assert_array_equal(shifted[COVARIATES],d[COVARIATES])
        self.assertEqual(DescriptorTransform('fraction').fit(d).transform(d).shape[1],31)

    def test_precursor_weighting(self):
        d = self.data
        x = DescriptorTransform('precursor').fit(d).transform(d)
        f = d.nominal_frac_element1
        for c in [v for v in x if v.startswith('precursor_weighted_')]:
            b = c.replace('precursor_weighted_','')
            np.testing.assert_allclose(x[c],f*x[b+'_element1']+(1-f)*x[b+'_element2'])

    def test_imputation_training_only(self):
        j = self.jobs[0]
        t = DescriptorTransform('icp').fit(j['train'])
        medians = dict(t.medians)
        altered = j['test'].copy()
        altered['work_function_element1'] = 999.
        t.transform(altered)
        self.assertEqual(t.medians,medians)

    def test_eta_targets_match(self):
        new = eta_targets(self.data).set_index('sample')
        saved = ROOT/'results/eta10_target_audit.csv'
        if saved.exists():
            expected = pd.read_csv(saved).set_index('sample')
            for c in ['included','original_only_target_mV','averaged_target_mV','n_accepted']:
                np.testing.assert_allclose(new[c],expected.loc[new.index,c],atol=1e-10,rtol=0,equal_nan=True)
        self.assertEqual(new.included.sum(),298)
        self.assertEqual(new.loc[new.included,'n_accepted'].sum(),711)

    def test_eta_folds_match_between_targets(self):
        for df in memberships([j for j in self.jobs if j['workflow']=='eta10']):
            a = df.loc[df.arm.eq('original_only')].drop(columns='arm').reset_index(drop=True)
            b = df.loc[df.arm.eq('averaged')].drop(columns='arm').reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b)

    def test_complete_linkage(self):
        r = np.random.default_rng(5)
        x = r.normal(size=(200,4))
        x[:,1] = -x[:,0]
        groups, corr = correlation_groups(x,.9)
        self.assertTrue(any(g==[0,1] for g in groups))
        for g in groups:
            self.assertGreaterEqual(np.min(np.abs(corr[np.ix_(g,g)])),.9-1e-10)

    def test_best_not_compact(self):
        candidates = [dict(mean=.55,sd=.03,k=20,model='RF',config_id=0),
                      dict(mean=.54,sd=.02,k=10,model='RF',config_id=0)]
        best, compact, _ = choose_candidates(candidates,3)
        self.assertEqual(best['k'],20)
        self.assertEqual(compact['k'],10)

    def test_estimator_determinism(self):
        x = np.random.default_rng(6).normal(size=(50,4))
        params = self.config['path_config']
        a = estimator('ET',params,8,42).fit(x,x[:,0])
        b = estimator('ET',params,8,42).fit(x,x[:,0])
        np.testing.assert_array_equal(a.predict(x),b.predict(x))

    def test_selector_reproducibility(self):
        j = self.jobs[0]
        config = dict(self.config,selection_trees=3)
        selection = dict(resamples=4,resample_train_fraction=.8,ranking_permutations=2)
        a = fit_selector(j['train'],j['protocol'],.9,config,selection,42,'icp')
        b = fit_selector(j['train'],j['protocol'],.9,config,selection,42,'icp')
        self.assertEqual(a,b)
        self.assertEqual(len(a['ranked']),len(a['groups']))
        for g in a['groups']:
            self.assertEqual(len(set(g)&set(a['ranked'])),1)

    def test_resamples_no_overlap(self):
        for protocol in ['stratified_random','unseen_pair']:
            j = next(j for j in self.jobs if j['protocol']==protocol and j['arm']=='with_bo')
            for a,b in resample_splits(j['train'],protocol,3,.8,42):
                self.assertFalse(set(a['sample'])&set(b['sample']))
                if protocol=='unseen_pair':
                    self.assertFalse(set(a.metal_pair)&set(b.metal_pair))


if __name__=='__main__':
    unittest.main()
