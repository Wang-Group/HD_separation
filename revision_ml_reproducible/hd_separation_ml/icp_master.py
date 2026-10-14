"""Read and validate the authoritative ICP workbook without stale formula caches."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "outputs/icp_master_20261012/ICP_master_20261012.xlsx"
SCHEMA_VERSION = "icp-master-1"
ATOMIC_MASS = {
    "Ag": 107.8682, "Al": 26.9815385, "Ba": 137.327, "Ca": 40.078,
    "Cd": 112.414, "Ce": 140.116, "Co": 58.933194, "Cu": 63.546, "Fe": 55.845,
    "La": 138.90547, "Li": 6.94, "Mg": 24.305, "Mn": 54.938044,
    "Mo": 95.95, "Na": 22.98976928, "Ni": 58.6934, "Pb": 207.2,
    "Pt": 195.084, "Ru": 101.07, "Si": 28.085, "V": 50.9415,
    "Zn": 65.38, "Zr": 91.224, "Cr": 51.9961, "K": 39.0983,
    "Ti": 47.867, "W": 183.84, "Y": 88.90584, "Sc": 44.955908, "Eu": 151.964,
    "Ga": 69.723, "In": 114.818, "Sr": 87.62, "Pd": 106.42, "Nb": 92.90637,
}

RAW_COLUMNS = [
    "raw_sample", "M1_raw", "M2_raw", "record_id", "M1", "M2", "raw_unit",
    "source_cohort", "source_file", "source_row", "received_molar1", "received_molar2",
    "received_atomic_mass1", "received_atomic_mass2", "received_amount1", "received_amount2",
    "amount_factor_mL", "received_molar_fraction1", "received_molar_fraction2",
    "atomic_mass1", "atomic_mass2", "nonnegative_mass1", "nonnegative_mass2",
    "molar_proxy1", "molar_proxy2", "molar_fraction1", "molar_fraction2",
    "mass_fraction1", "mass_fraction2", "amount1_umol", "amount2_umol", "measurement_status",
    "received_amount_formula1", "received_amount_formula2", "source_notes",
]
SAMPLE_COLUMNS = [
    "sample", "M1_raw", "M2_raw", "element1", "element2", "molar_fraction1",
    "molar_fraction2", "mass_fraction1", "mass_fraction2", "cohort", "include_in_ml",
    "status", "separation_coefficient", "measurement_id", "shared_icp_group",
    "nominal_fraction1", "nominal_fraction2", "base_policy", "source_record_id",
    "raw_unit", "amount1_umol", "amount2_umol", "notes",
]
ELIGIBLE = {"pure_assumed_100", "measured_valid", "one_component_zero_accepted"}


def is_number(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(value)


def calculate_measurement(record, masses):
    result = {}
    for side in (1, 2):
        mass = masses.get(record[f"M{side}"])
        value = record[f"M{side}_raw"]
        positive = max(float(value), 0) if is_number(value) else None
        result[f"atomic_mass{side}"] = mass
        result[f"nonnegative_mass{side}"] = positive
        result[f"molar_proxy{side}"] = positive / mass if positive is not None and mass else None
        factor = record.get("amount_factor_mL")
        molar = result[f"molar_proxy{side}"]
        result[f"amount{side}_umol"] = molar * factor if molar is not None and is_number(factor) else None
    for prefix, numerator in (("molar", "molar_proxy"), ("mass", "nonnegative_mass")):
        a, b = result[f"{numerator}1"], result[f"{numerator}2"]
        valid = a is not None and b is not None and a + b > 0
        result[f"{prefix}_fraction1"] = a / (a + b) if valid else None
        result[f"{prefix}_fraction2"] = b / (a + b) if valid else None
    a, b = result["molar_proxy1"], result["molar_proxy2"]
    if a is None or b is None:
        status = "missing_raw_or_atomic_mass"
    elif a + b == 0:
        status = "both_zero"
    elif min(a, b) == 0:
        status = "one_component_zero"
    else:
        status = "valid"
    result["measurement_status"] = status
    return result


def sample_results(sample, measurement):
    result = {}
    for key in ("M1_raw", "M2_raw", "raw_unit", "amount1_umol", "amount2_umol"):
        result[key] = measurement.get(key) if measurement else None
    for prefix in ("molar", "mass"):
        for side in (1, 2):
            key = f"{prefix}_fraction{side}"
            result[key] = (1.0 if side == 1 else 0.0) if sample["base_policy"] == "pure" else (
                measurement.get(key) if measurement else None)
    policy = sample["base_policy"]
    if policy == "pure":
        result["status"] = "pure_assumed_100"
        result["amount1_umol"] = result["amount2_umol"] = None
    elif policy == "shared_reference":
        result["status"] = "excluded_shared_reference"
    elif policy == "reference_repeat":
        result["status"] = "excluded_reference_repeat"
    elif not measurement:
        result["status"] = "excluded_no_icp"
    else:
        result["status"] = {
            "valid": "measured_valid", "one_component_zero": "one_component_zero_accepted",
            "both_zero": "excluded_both_zero", "missing_raw_or_atomic_mass": "excluded_no_icp",
        }[measurement["measurement_status"]]
    result["include_in_ml"] = result["status"] in ELIGIBLE
    return result


def _rows(sheet, columns):
    rows = list(sheet.iter_rows(min_row=6, max_col=len(columns), values_only=True))
    if list(rows[0]) != columns:
        raise ValueError(f"Unexpected columns in {sheet.title}; expected master schema {SCHEMA_VERSION}.")
    return [dict(zip(columns, row)) for row in rows[1:] if row[0] is not None]


def _same(actual, expected):
    if expected is None:
        return actual is None or actual == ""
    if isinstance(expected, bool):
        return actual is expected
    if is_number(expected):
        return is_number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    return actual == expected


def load_master(path=MASTER, *, verify_cached=True):
    """Recompute from workbook raw inputs; reject stale displayed results by default."""
    book = load_workbook(path, data_only=True, read_only=True)
    try:
        if book["Overview"]["B3"].value != SCHEMA_VERSION:
            raise ValueError("Unsupported ICP workbook schema.")
        masses = dict(book["AtomicMass"].iter_rows(min_row=7, max_col=2, values_only=True))
        if set(masses) != set(ATOMIC_MASS) or any(not _same(masses[e], a) for e, a in ATOMIC_MASS.items()):
            raise ValueError("AtomicMass differs from the user-approved dictionary; version the change first.")
        raw_rows = _rows(book["RawICP"], RAW_COLUMNS)
        samples = _rows(book["Samples"], SAMPLE_COLUMNS)
        measurements = {}
        for row in raw_rows:
            if row["record_id"] in measurements:
                raise ValueError(f"Duplicate ICP record: {row['record_id']}")
            computed = calculate_measurement(row, masses)
            if verify_cached:
                for key, value in computed.items():
                    if not _same(row[key], value):
                        raise ValueError(f"Stale/incorrect RawICP cache: {row['record_id']} {key}; rebuild or recalculate Excel.")
            measurements[row["record_id"]] = {**row, **computed}
        seen = set()
        for sample in samples:
            if sample["sample"] in seen:
                raise ValueError(f"Duplicate sample: {sample['sample']}")
            seen.add(sample["sample"])
            if sample["cohort"] == "BO" and sample["sample"].startswith("Co-Fe"):
                raise ValueError("Co-Fe BO is excluded by the confirmed policy.")
            mid = sample["measurement_id"]
            if mid and mid not in measurements:
                raise ValueError(f"Unknown measurement ID: {mid}")
            measurement = measurements.get(mid)
            if measurement and (sample["element1"], sample["element2"]) != (measurement["M1"], measurement["M2"]):
                raise ValueError(f"Element order mismatch for {sample['sample']}")
            computed = sample_results(sample, measurement)
            if verify_cached:
                for key, value in computed.items():
                    if not _same(sample[key], value):
                        raise ValueError(f"Stale/incorrect Samples cache: {sample['sample']} {key}; rebuild or recalculate Excel.")
            sample.update(computed)
        return pd.DataFrame(samples), pd.DataFrame(measurements.values())
    finally:
        book.close()


def load_initial_composition(path=MASTER, *, eligible_only=True):
    samples, _ = load_master(path)
    frame = samples[samples.cohort.eq("initial")].copy()
    if eligible_only:
        frame = frame[frame.include_in_ml].copy()
    frame["merge_status"] = frame.status.map({
        "pure_assumed_100": "single_assumed_100", "measured_valid": "measured_valid",
        "one_component_zero_accepted": "measured_valid", "excluded_both_zero": "excluded_no_detectable_metals",
        "excluded_no_icp": "excluded_au_not_measured",
    })
    frame["include_in_actual_analysis"] = frame.include_in_ml
    frame["matched_zero_component_accepted"] = frame.status.eq("one_component_zero_accepted")
    frame["matched_preferred_true_used"] = frame["sample"].eq("V-Co")
    return frame.rename(columns={
        "molar_fraction1": "actual_frac_element1", "molar_fraction2": "actual_frac_element2",
        "nominal_fraction1": "nominal_frac_element1", "nominal_fraction2": "nominal_frac_element2",
    }).reset_index(drop=True)


def load_bo_composition(path=MASTER):
    samples, raw = load_master(path)
    frame = samples[samples.cohort.eq("BO") & samples.measurement_id.notna()].copy()
    by_id = raw.set_index("record_id")
    frame["icp_raw_sample"] = frame.measurement_id.map(by_id.raw_sample)
    for out, source in (
        ("icp_m1_mass_ug_ml", "M1_raw"), ("icp_m2_mass_ug_ml", "M2_raw"),
        ("icp_n1_umol", "amount1_umol"), ("icp_n2_umol", "amount2_umol"),
    ):
        frame[out] = frame[source]
    for side in (1, 2):
        frame[f"icp_m{side}_umol_ml"] = frame.measurement_id.map(by_id[f"molar_proxy{side}"])
        frame[f"icp_m{side}_mmol_ml"] = frame[f"icp_m{side}_umol_ml"] / 1000
    frame["icp_total_metal_umol"] = frame.icp_n1_umol + frame.icp_n2_umol
    frame["icp_c1_c2_molar_ratio"] = frame.icp_m1_umol_ml / frame.icp_m2_umol_ml
    frame["icp_n1_n2_molar_ratio"] = frame.icp_n1_umol / frame.icp_n2_umol
    frame["icp_w1_w2_mass_ratio"] = frame.mass_fraction1 / frame.mass_fraction2
    frame["icp_deposited_m1_m2_molar_ratio"] = frame.icp_c1_c2_molar_ratio
    frame["icp_ratio_source"] = "master raw mass / approved atomic mass"
    frame["actual_frac1_from_icp"] = frame.molar_fraction1
    frame["actual_frac1_from_icp_amount"] = frame.icp_n1_umol / frame.icp_total_metal_umol
    frame["icp_source_columns"] = SCHEMA_VERSION
    columns = ["sample", "actual_frac1_from_icp", "actual_frac1_from_icp_amount"]
    return frame[columns + [c for c in frame if c.startswith("icp_")]].reset_index(drop=True)


if __name__ == "__main__":
    import json
    data, raw = load_master()
    print(json.dumps({"samples": len(data), "raw_records": len(raw),
                      "included": int(data.include_in_ml.sum()),
                      "status": data.status.value_counts().to_dict()}, indent=2))
