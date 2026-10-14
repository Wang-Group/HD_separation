"""Focused tests for the four reviewer follow-up analyses."""
import unittest

import numpy as np
import pandas as pd

from revision_followup_common import BASES,FEATURES,representation,clean_inputs,aggregate_predictions
from revision_correlations import observed_features,shifted_features,correlations,COVARIATES
from revision_operating_ablation import inputs
from revision_missing_references import eligibility
from revision_shap_current import aggregate


def fixture():
    data=pd.DataFrame({"sample":["a","b","c"],"element1":["Co","Nb","Co"],
        "element2":["Ni","Ni","Eu"],"actual_frac_element1":[.25,0.,1.],
        "separation_coefficient":[4.,7.,8.],"potential_offset":[-.2,.1,.3]})
    for b in BASES:
        data[f"{b}_element1"]=[2.,np.nan,2.]
        data[f"{b}_element2"]=[4.,4.,np.nan]
    for c in COVARIATES:
        data[c]=[1.,2.,3.]
    return data


class FollowupTests(unittest.TestCase):
    def test_observed_weighted_zero_component(self):
        x=observed_features(fixture())
        np.testing.assert_allclose(x.icp_weighted_work_function,[3.5,4.,2.])
        self.assertTrue(np.isnan(x.work_function_element1.iloc[1]))

    def test_missing_nonzero_weight_remains_missing(self):
        d=fixture(); d.loc[1,"actual_frac_element1"]=.1
        self.assertTrue(np.isnan(observed_features(d).icp_weighted_work_function.iloc[1]))

    def test_selected_complete_does_not_require_unselected(self):
        d=fixture()
        np.testing.assert_array_equal(eligibility(d,["icp_weighted_work_function"],"selected_properties_complete",set()),[True,True,True])
        np.testing.assert_array_equal(eligibility(d,["work_function_element1"],"selected_properties_complete",set()),[True,False,True])

    def test_unlisted_policy_checks_both_elements(self):
        np.testing.assert_array_equal(eligibility(fixture(),[],"listed_elements_only",{"Co","Ni"}),[True,False,False])

    def test_shift_equations(self):
        d=fixture(); x=observed_features(d); shifted=shifted_features(x,d.potential_offset)
        for b,sign in (("work_function",1),("pzc",-1),("E_std",-1)):
            for c in (b+"_element1",b+"_element2","icp_weighted_"+b):
                np.testing.assert_allclose(shifted[c],x[c]+sign*d.potential_offset,equal_nan=True)
        np.testing.assert_allclose(shifted.delta_EOH_element1,x.delta_EOH_element1,equal_nan=True)
        self.assertNotEqual(shifted.iloc[0,0],x.iloc[0,0])

    def test_operating_inclusion_allowlist(self):
        d=fixture(); transform=representation("hybrid39").fit(clean_inputs(d))
        baseline=inputs(transform,d,"intrinsic39")
        self.assertEqual(list(baseline.columns),FEATURES)
        poisoned=d.copy(); poisoned[COVARIATES]=1e9; poisoned["separation_coefficient"]=-1e9
        np.testing.assert_allclose(baseline,inputs(transform,poisoned,"intrinsic39"))
        expanded=inputs(transform,d,"intrinsic39_operating")
        self.assertEqual(list(expanded.columns),FEATURES+COVARIATES)

    def test_imputation_uses_training_only(self):
        d=fixture(); transform=representation("hybrid39").fit(clean_inputs(d.iloc[:1]))
        d.loc[1,"work_function_element2"]=1000
        self.assertEqual(inputs(transform,d,"intrinsic39").work_function_element1.iloc[1],3.)

    def test_pairwise_correlations_and_constant(self):
        d=pd.DataFrame(dict(a=[1.,2.,3.,np.nan],b=[2.,4.,6.,8.],constant=[1.,1.,1.,1.]))
        r=correlations(d,list(d.columns))
        pair=r[r.feature_a.eq("a")&r.feature_b.eq("b")]
        np.testing.assert_allclose(pair.r,1.)
        self.assertTrue(pair.n.eq(3).all())
        self.assertTrue(r[r.feature_b.eq("constant")].r.isna().all())

    def test_shap_aggregate_sample_weighted(self):
        d=pd.DataFrame([dict(protocol="a",kind="b",repeat=0,feature="f",n=2,abs_sum=2),
                        dict(protocol="a",kind="b",repeat=0,feature="f",n=8,abs_sum=24)])
        repeat,summary=aggregate(d,"feature")
        self.assertAlmostEqual(repeat.mean_abs_shap.iloc[0],2.6)
        self.assertAlmostEqual(summary.mean_abs_shap.iloc[0],2.6)

    def test_duplicate_sample_in_repeat_rejected(self):
        d=pd.DataFrame(dict(case=["a","a"],repeat=[0,0],sample=["same","same"],y_true=[1,2],y_pred=[1,2]))
        with self.assertRaises(ValueError):
            aggregate_predictions(d,["case"])


if __name__=="__main__":
    unittest.main()
