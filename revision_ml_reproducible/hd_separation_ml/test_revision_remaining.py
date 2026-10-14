"""Focused checks for matched composition, SHAP aggregation and shape analysis."""
import json
import unittest

import numpy as np
import pandas as pd

from revision_remaining_common import BASE11, POOL, add_recipes, reweight, observed_weighted, RECIPE_COFE
from revision_composition_bridge33 import feature_frame, matched_inner
from revision_electronegativity_branches import quadratic, bootstrap_indices, binned_summary
from revision_shap33 import aggregate


def fixture():
    d=pd.DataFrame(dict(sample=["a","b","c"],actual_frac_element1=[.2,0.,1.],
        nominal_frac_element1=[.8,.5,.6],separation_coefficient=[3.,5.,7.]))
    x=pd.DataFrame(index=d.index)
    for b in BASE11:
        d[b+"_element1"]=[2.,np.nan,2.]
        d[b+"_element2"]=[4.,4.,np.nan]
        x[b+"_element1"]=[2.,3.,2.]
        x[b+"_element2"]=[4.,4.,3.]
        x["icp_weighted_"+b]=[3.6,4.,2.]
    return d,x[POOL]


class RemainingTests(unittest.TestCase):
    def test_only_weights_change(self):
        d,x=fixture(); before=x.copy()
        z=reweight(x,d,"precursor")
        np.testing.assert_allclose(z.icp_weighted_work_function,[2.4,3.5,2.4])
        pd.testing.assert_frame_equal(x,before)
        cols=[c for c in POOL if not c.startswith("icp_weighted_")]
        pd.testing.assert_frame_equal(z[cols],x[cols])

    def test_weighting_does_not_align_mismatched_indices(self):
        d,x=fixture(); d.index=["a","b","c"]
        np.testing.assert_allclose(reweight(x,d,"precursor").icp_weighted_work_function,[2.4,3.5,2.4])

    def test_observed_zero_and_missing_components(self):
        d,_=fixture()
        np.testing.assert_allclose(observed_weighted(d,"work_function","icp"),[3.6,4.,2.])
        self.assertTrue(np.isnan(observed_weighted(d,"work_function","precursor")[1:]).all())

    def test_expanded_inputs_are_allowlisted(self):
        d,x=fixture()
        self.assertEqual(feature_frame(x,d,"raw22").shape[1],22)
        self.assertEqual(feature_frame(x,d,"precursor33").shape[1],33)
        z=feature_frame(x,d,"icp33_plus_fraction")
        self.assertEqual(z.shape[1],34)
        d["separation_coefficient"]=-1000
        pd.testing.assert_frame_equal(z,feature_frame(x,d,"icp33_plus_fraction"))

    def test_inner_membership_order_is_verified(self):
        d,x=fixture(); d=d.set_index("sample",drop=False)
        inner=[dict(X=x.iloc[:2].to_numpy(),V=x.iloc[2:].to_numpy(),y=np.array([3.,5.]),z=np.array([7.]))]
        members=[dict(inner_fold=0,role="train",sample="a"),dict(inner_fold=0,role="train",sample="b"),
                 dict(inner_fold=0,role="validation",sample="c")]
        out=matched_inner(inner,members,d,"precursor33")[0]
        col=out["columns"].index("icp_weighted_work_function")
        np.testing.assert_allclose(out["X"][:,col],[2.4,3.5])
        with self.assertRaises(AssertionError):
            matched_inner(inner,[members[1],members[0],members[2]],d,"precursor33")

    def test_recipe_target_mismatch_is_rejected(self):
        source=pd.DataFrame(json.loads(RECIPE_COFE.read_text(encoding="utf-8")))[0:1].iloc[0].to_dict()
        d=pd.DataFrame([dict(sample="Co-Fe0",cohort="BO",element1="Co",element2="Fe",
            actual_frac_element1=.4,actual_frac_element2=.6,nominal_frac_element1=np.nan,
            nominal_frac_element2=np.nan,separation_coefficient=source["separation_coefficient"])]).set_index("sample",drop=False)
        result=add_recipes(d)
        self.assertAlmostEqual(result.nominal_frac_element1.iloc[0],source["volume1"]/(source["volume1"]+source["volume2"]))
        d["separation_coefficient"]+=1
        with self.assertRaises(AssertionError): add_recipes(d)

    def test_quadratic_does_not_force_volcano(self):
        x=np.linspace(1,2,30)
        a=quadratic(x,10-3*(x-1.5)**2)
        self.assertTrue(a["interior_maximum"])
        self.assertAlmostEqual(a["vertex"],1.5)
        self.assertAlmostEqual(a["descriptive_r2"],1.)
        self.assertFalse(quadratic(x,3*(x-1.5)**2)["interior_maximum"])
        self.assertFalse(quadratic(x,10-(x-3)**2)["interior_maximum"])
        self.assertIsNone(quadratic(np.ones(30),x))

    def test_cluster_resampling_preserves_whole_pairs(self):
        groups=np.array(["a","a","b","b","b"])
        index=bootstrap_indices(groups,np.random.default_rng(4),"pair_cluster")
        self.assertEqual(np.sum(index==0),np.sum(index==1))
        self.assertEqual(np.sum(index==2),np.sum(index==3))
        self.assertEqual(np.sum(index==3),np.sum(index==4))

    def test_within_pair_resampling_retains_pair_counts(self):
        groups=np.array(["a","a","b","b","b"])
        index=bootstrap_indices(groups,np.random.default_rng(4),"within_pair")
        self.assertEqual(np.sum(groups[index]=="a"),2)
        self.assertEqual(np.sum(groups[index]=="b"),3)

    def test_bins_count_each_row_once(self):
        d=pd.DataFrame(dict(x=np.repeat(np.arange(5),6),separation_coefficient=np.arange(30),metal_pair="a"))
        b=binned_summary(d,"x",5,1)
        self.assertEqual(b.n.sum(),len(d))
        self.assertTrue((b.q90>=b["mean"]).all())
        self.assertTrue(binned_summary(d,"x",5,100).empty)

    def test_shap_uses_sample_weights_and_keeps_arms_distinct(self):
        rows=[dict(arm="with_bo",protocol="a",kind="b",repeat=0,feature="f",n=2,abs_sum=2),
              dict(arm="with_bo",protocol="a",kind="b",repeat=0,feature="f",n=8,abs_sum=24),
              dict(arm="without_bo",protocol="a",kind="b",repeat=0,feature="f",n=10,abs_sum=10)]
        _,s=aggregate(pd.DataFrame(rows),"feature")
        s=s.set_index("arm")
        self.assertAlmostEqual(s.loc["with_bo","mean_abs_shap"],2.6)
        self.assertAlmostEqual(s.loc["without_bo","mean_abs_shap"],1.)


if __name__=="__main__":
    unittest.main()
