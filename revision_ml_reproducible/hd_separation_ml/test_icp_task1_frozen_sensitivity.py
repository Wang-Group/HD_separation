import unittest

import numpy as np
import pandas as pd

from icp_task1_frozen_sensitivity import (
    FEATURES, TASK1, choose_global_config, consensus_features, fit_frozen, metric_tables,
)


def source_tables():
    settings,paths = [],[]
    for repeat in range(3):
        for fold in range(5):
            key = dict(arm="with_bo",protocol=TASK1,repeat=repeat,fold=fold)
            settings.append(dict(**key,strategy="forward10",features=FEATURES[:10]))
            paths.extend(dict(**key,n_features=i+1,added_feature=c) for i,c in enumerate(FEATURES[:16]))
    return pd.DataFrame(settings),pd.DataFrame(paths)


class FrozenSpecificationTests(unittest.TestCase):
    def test_consensus_uses_exact_feature_variants(self):
        settings,paths = source_tables()
        selected,ranking = consensus_features(settings,paths)
        self.assertEqual(selected,FEATURES[:10])
        self.assertTrue(ranking.head(10).source_selected_fits.eq(15).all())

    def test_other_tasks_cannot_change_selection(self):
        settings,paths = source_tables()
        extra = settings.copy()
        extra["protocol"] = "known_pair_new_composition"
        extra["features"] = [FEATURES[-10:] for _ in range(len(extra))]
        selected,_ = consensus_features(pd.concat([settings,extra]),paths)
        self.assertEqual(selected,FEATURES[:10])

    def test_missing_or_mismatched_paths_rejected(self):
        settings,paths = source_tables()
        with self.assertRaises(ValueError):
            consensus_features(settings,paths.iloc[1:])
        settings.at[0,"features"] = FEATURES[1:11]
        with self.assertRaises(ValueError):
            consensus_features(settings,paths)

    def test_config_choice_ignores_non_source_tasks_by_rejecting_them(self):
        rows = [dict(arm="with_bo",protocol=TASK1,model=m,config_id=0,repeat=r,fold=f,
            inner_fold=i,r2=.3 if m=="RF" else .4) for m in ("RF","ET")
            for r in range(3) for f in range(5) for i in range(3)]
        scores = pd.DataFrame(rows)
        chosen,_ = choose_global_config(scores,[{}])
        self.assertEqual(chosen["model"],"ET")
        scores.loc[0,"protocol"] = "unseen_pair"
        with self.assertRaises(ValueError):
            choose_global_config(scores,[{}])

    def test_metric_table_pools_folds_before_scoring(self):
        rows = [dict(arm="with_bo",protocol=TASK1,strategy="task1_frozen10",repeat=r,fold=i//2,
            sample=str(i),y_true=float(i),y_pred=float(i)+.5) for r in range(3) for i in range(4)]
        repeats,summary = metric_tables(pd.DataFrame(rows))
        np.testing.assert_allclose(repeats.r2,.8)
        self.assertEqual(summary.iloc[0].n,4)
        with self.assertRaises(ValueError):
            metric_tables(pd.DataFrame(rows+[rows[0]]))

    def test_test_response_and_forbidden_inputs_do_not_change_refit(self):
        from icp_electrochemical_exclusion import RAW
        rng = np.random.default_rng(8)
        train = pd.DataFrame(rng.normal(size=(25,len(RAW))),columns=RAW)
        train["actual_frac_element1"] = rng.uniform(0,1,25)
        train["separation_coefficient"] = rng.normal(size=25)
        test = train.iloc[:4].copy()
        test.loc[test.index[0],RAW[0]] = np.nan
        spec = dict(features=[FEATURES[0],FEATURES[-1]],model="RF",trees=8,
            config=dict(max_depth=3,min_samples_leaf=1,max_features=1.0))
        a = fit_frozen(train,test,spec,42)
        test["separation_coefficient"] = 1e9
        test["I_raw"] = -1e6
        test["eta10"] = 1e6
        b = fit_frozen(train,test,spec,42)
        np.testing.assert_array_equal(a,b)


if __name__ == "__main__":
    unittest.main()
