"""Validated English ICP inputs with independently replaceable response labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from icp_master import ATOMIC_MASS, calculate_measurement
from icp_intrinsic_descriptor_analysis import BASES, E_STD

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = ROOT / "revision_inputs"
WORKBOOK = REPO / "outputs/icp_simple_20261012/ICP_raw_and_percentages_EN.xlsx"
TARGETS = DATA / "targets.csv"
METADATA = DATA / "sample_metadata.csv"
PROPERTIES = DATA / "elemental_properties.csv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs():
    """One-time extraction; never overwrite a target file that a user may edit."""
    paths = [TARGETS, METADATA, PROPERTIES, DATA / "provenance.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError("Revision inputs already exist. Preserve edited targets; use a new input directory/version.")
    initial_path = ROOT / "outputs/feature_selection_561_rs_from_eis_no_eis_fitting/feature_matrix_561_no_eis_fitting.csv"
    bo_path = REPO / "0-Figures/Figure4/curated/figure4_bo_complete_112.csv"
    property_path = ROOT / "EH_extra.xlsx"
    initial = pd.read_csv(initial_path)
    bo = pd.read_csv(bo_path)
    bo["sample"] = bo.system + bo.row_id.astype(str)
    targets = pd.concat([initial[["sample", "separation_coefficient"]],
                         bo[["sample", "separation_coefficient"]]], ignore_index=True)
    if targets["sample"].duplicated().any():
        raise ValueError("Duplicate target IDs")
    metadata = initial[["sample", "I_raw", "E_IR_corrected_final"]].copy()
    metadata["nominal_frac_element1"] = np.where(initial.element1.eq(initial.element2), 1.0, 6 / 7)
    extra = bo[["sample"]].copy()
    extra["nominal_frac_element1"] = bo.volume1 / (bo.volume1 + bo.volume2)
    metadata = pd.concat([metadata, extra], ignore_index=True)
    properties = pd.read_excel(property_path, sheet_name="Sheet1").rename(columns={
        "Unnamed: 0": "element", "Work function (eV)": "work_function", "M-M": "MM_bond",
        "E(M-H) (kJ/mol)": "MH_bond_kJ", "E(M-D) (kJ/mol)": "MD_bond_kJ",
        "delta_ZPE_H (eV)": "delta_ZPE_H",
    })
    adsorption = pd.read_excel(property_path, sheet_name="Sheet2").rename(columns={"Unnamed: 0": "element"})
    adsorption["element"] = adsorption.element.ffill()
    properties["FE_ads_avg"] = properties.element.map(adsorption.groupby("element")["Free Energy of Adsorption (eV)"].mean())
    properties["E_std"] = properties.element.map(E_STD)
    properties = properties[["element", *BASES]]
    if properties.element.duplicated().any():
        raise ValueError("Duplicate elemental property IDs")
    DATA.mkdir(parents=True, exist_ok=True)
    targets.to_csv(TARGETS, index=False)
    metadata.to_csv(METADATA, index=False)
    properties.to_csv(PROPERTIES, index=False)
    provenance = {"source_hashes": {str(p.relative_to(REPO)): sha(p) for p in (initial_path, bo_path, property_path)},
                  "target": "separation_coefficient", "target_unit": "dimensionless",
                  "initial_nominal_fraction": "pure=1; bimetal M1=6/7, M2=1/7",
                  "properties": "Unshifted intrinsic elemental properties; no sample-potential correction.",
                  "n_targets": len(targets)}
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Prepared {len(targets)} target rows in {DATA}")


def read_composition(path=WORKBOOK):
    frame = pd.read_excel(path, sheet_name="ICP")
    rename = {"Sample": "sample", "M1": "element1", "M2": "element2", "Dataset": "cohort",
              "M1 molar percentage": "actual_frac_element1", "M2 molar percentage": "actual_frac_element2",
              "Shared ICP ID": "shared_icp_id", "Composition status": "composition_status"}
    required = [*rename, "M1 raw value", "M2 raw value", "M1 mass percentage", "M2 mass percentage",
                "M1 atomic mass", "M2 atomic mass", "Source record (file type / sample / row)"]
    if set(required) - set(frame):
        raise ValueError(f"Missing workbook columns: {set(required) - set(frame)}")
    frame = frame.rename(columns=rename)
    if frame["sample"].isna().any() or frame["sample"].duplicated().any():
        raise ValueError("Workbook sample IDs must be nonempty and unique")
    if not frame.cohort.isin(["Initial screening", "BO"]).all():
        raise ValueError("Unexpected Dataset values")
    frame["cohort"] = frame.cohort.replace({"Initial screening": "initial"})
    if ((frame.cohort == "BO") & frame["sample"].str.match(r"Co-Fe\d+$")).any():
        raise ValueError("Unexpected BO system in the confirmed analysis scope")
    for _, row in frame.iterrows():
        for side in (1, 2):
            element = row[f"element{side}"]
            mass = row[f"M{side} atomic mass"]
            expected_mass = ATOMIC_MASS.get(element, np.nan)
            if not np.isclose(mass, expected_mass, equal_nan=True, rtol=0, atol=1e-10):
                raise ValueError(f"Unapproved atomic mass: {row['sample']} {element}")
        if row.element1 == row.element2:
            expected = [1.0, 0.0, 1.0, 0.0]
        else:
            record = {"M1": row.element1, "M2": row.element2,
                      "M1_raw": row["M1 raw value"], "M2_raw": row["M2 raw value"]}
            values = calculate_measurement(record, ATOMIC_MASS)
            expected = [values[k] if values[k] is not None else np.nan for k in
                        ("molar_fraction1", "molar_fraction2", "mass_fraction1", "mass_fraction2")]
        found = [row.actual_frac_element1, row.actual_frac_element2,
                 row["M1 mass percentage"], row["M2 mass percentage"]]
        if not np.allclose(found, expected, equal_nan=True, rtol=0, atol=1e-10):
            raise ValueError(f"Stale or incorrect percentages for {row['sample']}. Recalculate and save the workbook.")
    frame["shared_icp_id"] = frame.shared_icp_id.fillna("")
    frame["icp_group"] = np.where(frame.shared_icp_id.ne(""), "shared:" + frame.shared_icp_id, "sample:" + frame["sample"])
    frame["shared_icp"] = frame.shared_icp_id.ne("")
    frame["zero_component"] = frame[["M1 raw value", "M2 raw value"]].lt(0).any(axis=1)
    for group, part in frame.groupby("icp_group"):
        if len(part) > 1:
            cols = ["element1", "element2", "actual_frac_element1", "actual_frac_element2", "M1 raw value", "M2 raw value"]
            if len(part[cols].drop_duplicates()) != 1:
                raise ValueError(f"Inconsistent shared ICP group: {group}")
    source = frame.loc[frame["sample"].eq("V-Co"), "Source record (file type / sample / row)"]
    if len(source) and not source.str.contains("V-Co-true", regex=False).all():
        raise ValueError("V-Co must trace to the confirmed V-Co-true measurement")
    return frame


def build_revision_data(workbook=WORKBOOK, targets=TARGETS, target="separation_coefficient",
                        metadata=METADATA, properties=PROPERTIES):
    frame = read_composition(workbook)
    labels = pd.read_csv(targets)
    if not {"sample", target} <= set(labels):
        raise ValueError(f"Target CSV must contain sample and {target!r}")
    if labels["sample"].isna().any() or labels["sample"].duplicated().any():
        raise ValueError("Target sample IDs must be nonempty and unique")
    unknown = set(labels["sample"]) - set(frame["sample"])
    if unknown:
        raise ValueError(f"Unknown target IDs: {sorted(unknown)}")
    labels = labels[["sample", target]].rename(columns={target: "separation_coefficient"})
    labels["separation_coefficient"] = pd.to_numeric(labels.separation_coefficient, errors="raise")
    if np.isinf(labels.separation_coefficient).any():
        raise ValueError("Infinite response labels")
    frame = frame.merge(labels, on="sample", how="left", validate="one_to_one")
    meta = pd.read_csv(metadata)
    if meta["sample"].duplicated().any():
        raise ValueError("Duplicate metadata IDs")
    frame = frame.merge(meta, on="sample", how="left", validate="one_to_one")
    if frame.nominal_frac_element1.isna().any() or not frame.nominal_frac_element1.between(0, 1).all():
        raise ValueError("Missing or invalid precursor fractions")
    frame["nominal_frac_element2"] = 1 - frame.nominal_frac_element1
    elemental = pd.read_csv(properties).set_index("element")
    if not elemental.index.is_unique or set(BASES) - set(elemental):
        raise ValueError("Invalid elemental-property table")
    for side in ("element1", "element2"):
        for base in BASES:
            frame[f"{base}_{side}"] = frame[side].map(elemental[base])
    columns = [f"{b}_{s}" for b in BASES for s in ("element1", "element2")]
    frame["properties_complete"] = frame[columns].notna().all(axis=1)
    frame["metal_pair"] = frame[["element1", "element2"]].apply(lambda r: "-".join(sorted(r)), axis=1)
    frame["status"] = "eligible"
    frame.loc[frame.separation_coefficient.isna(), "status"] = "missing_target"
    frame.loc[frame.actual_frac_element1.isna(), "status"] = "no_valid_icp_composition"
    # Select representatives independently of y: initial screening first, then input order.
    representative = frame.assign(priority=frame.cohort.ne("initial").astype(int)).sort_values("priority", kind="stable")
    representative = representative.drop_duplicates("icp_group")["sample"]
    frame.loc[~frame["sample"].isin(representative), "status"] = "shared_icp_not_independent"
    valid = frame[frame.status.eq("eligible")].copy()
    first = valid.metal_pair.str.split("-").str[0]
    fraction = valid.actual_frac_element1.where(valid.element1.eq(first), 1 - valid.actual_frac_element1)
    bins = np.minimum(np.floor((fraction.to_numpy() + 1e-12) / .05).astype(int), 19)
    valid["composition_group"] = valid.metal_pair + "_bin" + pd.Series(bins, index=valid.index).astype(str)
    if not np.isfinite(valid.separation_coefficient).all() or valid.separation_coefficient.nunique() < 3:
        raise ValueError("Insufficient finite, varying target values")
    initial = valid[valid.cohort.eq("initial")].reset_index(drop=True)
    bo = valid[valid.cohort.eq("BO")].reset_index(drop=True)
    if not set(bo.metal_pair) <= set(initial.metal_pair):
        raise ValueError("BO chemistries require an initial-screening counterpart for the selected protocols")
    return initial, bo, frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_inputs()
    a, b, audit = build_revision_data()
    print(json.dumps({"initial": len(a), "BO": len(b), "statuses": audit.status.value_counts().to_dict()}, indent=2))
