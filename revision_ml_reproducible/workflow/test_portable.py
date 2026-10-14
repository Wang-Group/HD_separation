"""Package-specific regression tests; run with python run.py test."""
import argparse
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from inputs import ROOT, REFERENCE, load_data, resolve_source, start_run, read_json
from modeling import POOL33, FEATURES, prepare, selected_settings
from main import inventory_check, keys_for


class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.audit=load_data()
        cls.assignments=pd.read_csv(REFERENCE / "outer_assignments.csv")

    def test_curated_cohort_is_rebuilt_from_workbooks(self):
        self.assertEqual(self.data.cohort.value_counts().to_dict(),{"initial":554,"BO":111})
        self.assertTrue(self.data.index.is_unique)
        self.assertEqual(self.data.loc[self.data.cohort.eq("BO"),"metal_pair"].value_counts().to_dict(),
            {"Co-Fe":27,"Co-Mo":21,"Co-Ni":21,"Co-V":21,"Fe-Ni":21})

    def test_exact_source_file_copies_and_paths(self):
        self.assertGreater(inventory_check(),100)
        records=read_json(ROOT / "source_inventory.json")["files"]
        for r in records:
            if r["role"] in ("input","raw_provenance"):
                self.assertEqual(r["path"],r["repository_relative_path"])
                self.assertFalse(Path(r["path"]).is_absolute())

    def test_all_three_representations_remove_only_two_properties(self):
        self.assertEqual(len(POOL33),33)
        self.assertEqual(len(set(FEATURES)-set(POOL33)),6)
        self.assertFalse(any("Frequency_D" in f or "delta_ZPE_H" in f for f in POOL33))
        self.assertEqual(sum("Frequency_H" in f for f in POOL33),3)

    def test_three_tasks_have_both_bo_arms(self):
        keys=keys_for(self.assignments)
        self.assertEqual(len(keys),90)
        self.assertEqual(len(set(keys)),90)

    def test_inner_memberships_match_saved_selection_for_each_task(self):
        saved=pd.read_csv(REFERENCE / "pool33/inner_assignments.csv")
        for protocol in ("stratified_random","unseen_pair","known_pair_new_composition"):
            for arm in ("without_bo","with_bo"):
                key=(arm,protocol,0,0)
                train,test,_,members,audit=prepare(key,self.data,self.assignments,33)
                self.assertFalse(set(train["sample"])&set(test["sample"]))
                old=saved[saved.arm.eq(arm)&saved.protocol.eq(protocol)&saved.repeat.eq(0)&saved.fold.eq(0)]
                cols=["inner_fold","role","sample"]
                pd.testing.assert_frame_equal(pd.DataFrame(members)[cols].reset_index(drop=True),old[cols].reset_index(drop=True))

    def test_saved_feature_lists_are_valid(self):
        for key in keys_for(self.assignments):
            choices=selected_settings(REFERENCE / "pool33",key)
            self.assertEqual(len(choices),4)
            selected=choices[choices.strategy.eq("forward10")].iloc[0].features
            self.assertEqual(len(selected),10)
            self.assertTrue(set(selected)<=set(POOL33))

    def test_model_sources_cannot_escape_package(self):
        with self.assertRaises(ValueError): resolve_source(ROOT.parent,33)

    def test_output_cannot_overwrite_input_snapshot(self):
        args=argparse.Namespace(command="train",output=ROOT / "hd_separation_ml/revision_inputs",jobs=1,targets=None,pool=33,source=None)
        with self.assertRaises(ValueError): start_run(args,self.data)

    def test_custom_y_changes_only_response(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            path=Path(directory) / "targets.csv"
            targets=self.data[["sample","separation_coefficient"]].copy()
            targets["separation_coefficient"]+=.25
            targets.to_csv(path,index=False)
            revised,_=load_data(path)
            np.testing.assert_allclose(revised.separation_coefficient,self.data.separation_coefficient+.25,atol=1e-12)
            pd.testing.assert_frame_equal(revised.drop(columns="separation_coefficient"),self.data.drop(columns="separation_coefficient"))
            targets.iloc[:-1].to_csv(path,index=False)
            with self.assertRaises(ValueError): load_data(path)


if __name__=="__main__":
    unittest.main()
