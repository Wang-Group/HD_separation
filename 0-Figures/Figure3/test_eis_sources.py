"""Check source identities without fitting EIS, DRT or ML models."""

import ast
import unittest
from pathlib import Path

import pandas as pd

import eis_hd_separation_validation as eis


class EISSourceTests(unittest.TestCase):
    def test_cohort_matches_saved_eis_and_drt(self):
        samples = pd.read_csv(eis.INITIAL_SAMPLE_FILE)["sample"]
        self.assertEqual(len(samples), 561)
        self.assertEqual(samples.nunique(), 561)
        expected = set(samples)
        self.assertEqual(eis.load_initial_samples(), expected)
        for relative in (
            "eis_hd_validation/eis_band_integral_features.csv",
            "eis_pydrttools_validation/drt_features.csv",
        ):
            with self.subTest(table=relative):
                saved = pd.read_csv(eis.FIG3_DIR / "data" / relative)
                self.assertEqual(set(saved["sample"]), expected)

    def test_raw_files_and_labels_cover_the_cohort(self):
        samples = eis.load_initial_samples()
        missing = [name for name in samples if not (eis.EIS_DIR / f"{name}.txt").is_file()]
        self.assertEqual(missing, [])
        self.assertTrue(samples.issubset(set(eis.load_labels()["sample"])))

    def test_drt_uses_the_same_cohort_without_loading_solver(self):
        source = eis.FIG3_DIR / "eis_drt_hd_separation_validation.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        wanted = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "INITIAL_SAMPLE_FILE"
                for target in node.targets
            ):
                wanted.append(node)
            if isinstance(node, ast.FunctionDef) and node.name == "load_initial_samples":
                wanted.append(node)
        self.assertEqual(len(wanted), 2)
        scope = {"FIG3_DIR": eis.FIG3_DIR, "pd": pd, "Path": Path}
        exec(compile(ast.Module(body=wanted, type_ignores=[]), str(source), "exec"), scope)
        self.assertEqual(scope["INITIAL_SAMPLE_FILE"], eis.INITIAL_SAMPLE_FILE)
        self.assertEqual(scope["load_initial_samples"](), eis.load_initial_samples())


if __name__ == "__main__":
    unittest.main()
