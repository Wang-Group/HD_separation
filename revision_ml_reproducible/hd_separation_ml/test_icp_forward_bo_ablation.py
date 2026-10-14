import unittest
from unittest.mock import patch

import pandas as pd

from icp_forward_bo_ablation import matched_plan, no_bo_inner, seed_for


def sample(name, cohort="initial", pair="Co-V", composition=None):
    return dict(sample=name, cohort=cohort, metal_pair=pair, icp_group=name,
                composition_group=composition or name, separation_coefficient=5.)


class BOAblationTests(unittest.TestCase):
    def test_reject_bo_before_constructing_inner_folds(self):
        with self.assertRaisesRegex(ValueError, "BO data entered"):
            list(no_bo_inner(pd.DataFrame([sample("b", "BO")]), "stratified_random", 1))

    def test_zero_bo_composition_task_uses_initial_grouped_proxy(self):
        initial = pd.DataFrame([sample("a", pair="Co-V"), sample("b", pair="Fe-Ni")])

        def partitions(original, added, protocol, augmented, seed):
            self.assertTrue(added.empty)
            self.assertFalse(augmented)
            self.assertEqual(protocol, "unseen_pair")
            yield original.iloc[:1], original.iloc[1:]

        with patch("icp_forward_bo_ablation.inner_partitions", side_effect=partitions):
            train, valid, task = list(no_bo_inner(initial, "known_pair_new_composition", 1))[0]
        self.assertEqual(task, "unseen_pair")
        self.assertFalse(set(train.metal_pair) & set(valid.metal_pair))

    def test_matched_test_and_initial_train_ignore_bo_outcomes(self):
        key = ("stratified_random", 0, 0)
        data = pd.DataFrame([sample("train"), sample("test"), sample("bo", "BO")]).set_index("sample", drop=False)
        assignments = pd.DataFrame([dict(protocol=key[0], repeat=0, fold=0, role=role, sample=name)
            for role, name in [("original", "train"), ("augmented", "bo"), ("testing", "test")]])
        a, augmented, test = matched_plan(key, data, assignments)
        changed = data.copy()
        changed.loc[changed.cohort.eq("BO"), "separation_coefficient"] = 1000000
        b, _, same_test = matched_plan(key, changed, assignments)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(test, same_test)
        self.assertEqual(len(augmented), 2)
        self.assertEqual(len(a), 1)

    def test_unseen_test_is_matched_initial_population(self):
        key = ("unseen_pair", 0, 0)
        rows = [sample("train", pair="Co-V"), sample("test", pair="Fe-Ni"),
                sample("train_bo", "BO", pair="Co-V"), sample("test_bo", "BO", pair="Fe-Ni")]
        data = pd.DataFrame(rows).set_index("sample", drop=False)
        assignments = pd.DataFrame([dict(protocol=key[0], repeat=0, fold=0, role=role, sample=name)
            for role, name in [("original", "train"), ("augmented", "train_bo"),
                               ("testing", "test"), ("testing", "test_bo")]])
        original, augmented, test = matched_plan(key, data, assignments)
        self.assertEqual(test["sample"].tolist(), ["test"])
        self.assertTrue(original.cohort.eq("initial").all())
        self.assertFalse(set(augmented.metal_pair) & set(test.metal_pair))

    def test_seeds_match_augmented_comparison(self):
        self.assertEqual(seed_for(("unseen_pair", 2, 4)), 2204)
        self.assertEqual(seed_for(("known_pair_new_composition", 2, 4)), 7204)
        self.assertEqual(seed_for(("stratified_random", 2, 4)), 7204)

    def test_paired_effect_direction_and_sd(self):
        from summarize_icp_forward_bo_ablation import paired_deltas
        rows = []
        for repeat, gain in enumerate([.1, .2, .3]):
            for arm in ["without_bo", "with_bo"]:
                value = gain if arm == "with_bo" else 0
                rows.append(dict(protocol="stratified_random", repeat=repeat, strategy="all39", arm=arm,
                                 r2=.4+value, rmse=2-value, mae=1-value, n=5))
        result, repeats = paired_deltas(pd.DataFrame(rows))
        self.assertAlmostEqual(result.bo_delta_r2_mean.iloc[0], .2)
        self.assertAlmostEqual(result.bo_delta_r2_sd.iloc[0], .1)
        self.assertAlmostEqual(result.bo_delta_rmse_mean.iloc[0], -.2)
        self.assertEqual(len(repeats), 3)

    def test_paired_effect_rejects_unmatched_repeats(self):
        from summarize_icp_forward_bo_ablation import paired_deltas
        rows = [dict(protocol="stratified_random", repeat=i, strategy="all39", arm=arm,
                     r2=.5, rmse=1, mae=.8, n=5)
                for i,arm in [(0,"without_bo"),(1,"with_bo")]]
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            paired_deltas(pd.DataFrame(rows))


if __name__ == "__main__":
    unittest.main()
