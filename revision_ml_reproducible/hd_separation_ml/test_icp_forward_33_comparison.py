"""Tests for a fixed descriptor-family ablation, not data-driven pruning."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import icp_forward_33_comparison as comparison
from revision_followup_common import FEATURES,BASES,representation,clean_inputs


class ReducedPoolTests(unittest.TestCase):
    def test_exact_removed_pool(self):
        self.assertEqual(len(comparison.POOL),33)
        self.assertEqual(set(comparison.REMOVED),{
            "Frequency_D_element1","Frequency_D_element2","delta_ZPE_H_element1",
            "delta_ZPE_H_element2","icp_weighted_Frequency_D","icp_weighted_delta_ZPE_H"})
        self.assertTrue({"Frequency_H_element1","Frequency_H_element2","icp_weighted_Frequency_H"}<=set(comparison.POOL))

    def test_preserved_column_order(self):
        indices=[FEATURES.index(c) for c in comparison.POOL]
        self.assertEqual(indices,sorted(indices))
        self.assertEqual(len(set(indices)),33)

    def test_retained_values_and_imputation_unchanged(self):
        d=pd.DataFrame(dict(actual_frac_element1=[.2,.6,.8]))
        for base in BASES:
            d[base+"_element1"]=[1.,2.,np.nan]
            d[base+"_element2"]=[3.,4.,5.]
        transform=representation("hybrid39").fit(clean_inputs(d.iloc[:2]))
        full=transform.transform(clean_inputs(d))
        reduced=full[comparison.POOL]
        self.assertEqual(reduced.shape,(3,33))
        np.testing.assert_allclose(reduced.icp_weighted_Frequency_H,[2.6,2.8,3.0])
        self.assertEqual(reduced.work_function_element1.iloc[2],2.5)

    def test_cache_key_is_order_sensitive(self):
        self.assertNotEqual(comparison.trial_key(dict(mask=[0,1])),comparison.trial_key(dict(mask=[1,0])))

    def test_only_identical_ordered_subsets_reused(self):
        first,second=comparison.POOL[:2]
        rows=lambda score:[dict(model=m,inner_fold=i,r2=score,rmse=1.,mae=1.) for m in ("RF","ET") for i in range(3)]
        old=dict(trials=[dict(mask=[FEATURES.index(first)],inner_scores=rows(9.)),
            dict(mask=[FEATURES.index(first),FEATURES.index(second)],inner_scores=rows(8.)),
            dict(mask=[FEATURES.index(comparison.REMOVED[0])],inner_scores=rows(100.))])
        def evaluate(inner,mask,name,config,trees,seed):
            return [dict(inner_fold=i,r2=-.1,rmse=1.,mae=1.) for i in range(3)]
        with tempfile.TemporaryDirectory() as temp,patch.object(comparison,"evaluate_subset",side_effect=evaluate) as mock:
            path=Path(temp)/"path.json"
            result=comparison.reduced_path([],dict(max_features=2,selection_trees=1),42,old,path,"test",Path(temp)/"progress.json")
            self.assertEqual(result["selected"],[0,1])
            self.assertEqual(len(result["trials"]),65)
            self.assertEqual(sum(t["reused_identical_subset"] for t in result["trials"]),2)
            self.assertEqual(mock.call_count,126)
            self.assertFalse(any(t["added_feature"] in comparison.REMOVED for t in result["trials"]))
            mock.reset_mock()
            comparison.reduced_path([],dict(max_features=2,selection_trees=1),42,old,path,"test",Path(temp)/"progress.json")
            mock.assert_not_called()
            with self.assertRaises(ValueError):
                comparison.reduced_path([],dict(max_features=2,selection_trees=1),42,old,path,"changed",Path(temp)/"progress.json")


class PairedReportTests(unittest.TestCase):
    @staticmethod
    def predictions():
        return pd.DataFrame([dict(pool=pool,arm="with_bo",protocol="stratified_random",strategy="forward10",
            repeat=r,fold=0,sample=str(i),y_true=float(i),y_pred=float(i if pool==33 or i<3 else 4))
            for pool in (39,33) for r in range(3) for i in (1,2,3)])

    def test_paired_repeat_metrics(self):
        from summarize_icp_33_comparison import compare
        _,summary,paired,delta=compare(self.predictions())
        np.testing.assert_allclose(summary.set_index("pool").loc[[39,33],"r2_mean"],[.5,1.])
        np.testing.assert_allclose(paired.r2_delta,.5)
        self.assertAlmostEqual(delta.r2_delta_mean.iloc[0],.5)

    def test_unmatched_sample_rejected(self):
        from summarize_icp_33_comparison import compare
        with self.assertRaises(AssertionError):
            compare(self.predictions().iloc[:-1])

    def test_changed_target_rejected(self):
        from summarize_icp_33_comparison import compare
        p=self.predictions(); p.loc[p.pool.eq(33),"y_true"]+=1
        with self.assertRaises(AssertionError):
            compare(p)


if __name__=="__main__":
    unittest.main()
