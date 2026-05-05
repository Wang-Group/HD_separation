#!/usr/bin/env python3
"""Audit curated all_data files against archived raw_data sources.

The audit combines two checks:

1. Expected mapping: infer the raw row file from each archive batch mapping
   JSON (substance1/substance2 and row id).
2. Exact hash provenance: find raw archive files whose SHA256 matches the
   curated all_data file exactly.

No source data are modified by this script.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ALL_DATA_DIR = ROOT / "hd_separation_ml" / "data_extraction" / "all_data"
RAW_ROOT = Path(
    r"C:\Users\yibinjiang\Desktop\corrected_data_source\0-data-archive\raw_data"
)
OUT_DIR = ROOT / "hd_separation_ml" / "data_extraction" / "all_data_raw_audit"

DATA_TYPES = {
    "CV20": {
        "current_dir": ALL_DATA_DIR / "CV20",
        "raw_parts": [("electrocatalysis", "CV", "20")],
    },
    "EIS": {
        "current_dir": ALL_DATA_DIR / "EIS",
        "raw_parts": [("electrocatalysis", "EIS")],
    },
    "Et": {
        "current_dir": ALL_DATA_DIR / "Et",
        "raw_parts": [("electrocatalysis", "Et")],
    },
    "LSV": {
        "current_dir": ALL_DATA_DIR / "LSV",
        "raw_parts": [
            ("electrocatalysis", "LSV", "2"),
            ("electrocatalysis", "LSV"),
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(RAW_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)


def numeric_row_id(raw_key: object) -> str:
    text = str(raw_key)
    try:
        value = float(text)
        if value.is_integer():
            return str(int(value))
    except ValueError:
        pass
    return text


def sample_name_from_mapping(mapping_file: Path, substance1: str, substance2: str, row_id: str) -> str:
    if "BO-raw-data" in mapping_file.parts:
        return f"{substance1}-{substance2}{row_id}"
    return f"{substance1}-{substance2}"


def electrocatalysis_dir_for_mapping(mapping_file: Path) -> Path:
    if mapping_file.parent.name.lower() == "tmp":
        return mapping_file.parent
    tmp_dir = mapping_file.parent / "tmp"
    if tmp_dir.exists():
        return tmp_dir
    return mapping_file.parent


def raw_paths_for(mapping_file: Path, data_type: str, row_id: str) -> list[Path]:
    base = electrocatalysis_dir_for_mapping(mapping_file)
    return [
        base.joinpath(*parts, f"{row_id}.txt")
        for parts in DATA_TYPES[data_type]["raw_parts"]
    ]


def load_mapping_json(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if "substance1" not in obj or "substance2" not in obj:
        return None
    if not isinstance(obj["substance1"], dict) or not isinstance(obj["substance2"], dict):
        return None
    return obj


def mapping_priority(path: Path) -> tuple[int, str]:
    """Prefer tmp mapping files over stale top-level copies."""
    is_tmp = path.parent.name.lower() == "tmp"
    return (0 if is_tmp else 1, str(path))


def build_expected_mapping() -> pd.DataFrame:
    rows = []
    mapping_files = sorted(RAW_ROOT.rglob("initial*.json"), key=mapping_priority)
    for mapping_file in mapping_files:
        obj = load_mapping_json(mapping_file)
        if obj is None:
            continue

        keys = sorted(
            set(obj["substance1"]).intersection(obj["substance2"]),
            key=lambda value: int(value) if str(value).isdigit() else str(value),
        )
        for key in keys:
            row_id = numeric_row_id(key)
            substance1 = str(obj["substance1"][key])
            substance2 = str(obj["substance2"][key])
            sample = sample_name_from_mapping(mapping_file, substance1, substance2, row_id)
            for data_type in DATA_TYPES:
                for expected in raw_paths_for(mapping_file, data_type, row_id):
                    rows.append(
                        {
                            "sample": sample,
                            "data_type": data_type,
                            "substance1": substance1,
                            "substance2": substance2,
                            "row_id": row_id,
                            "mapping_file": str(mapping_file),
                            "expected_raw_path": str(expected),
                            "expected_raw_relpath": rel(expected),
                            "expected_raw_exists": expected.exists(),
                            "expected_raw_size": expected.stat().st_size if expected.exists() else pd.NA,
                        }
                    )
    mapping = pd.DataFrame(rows).drop_duplicates(
        subset=["sample", "data_type", "expected_raw_path"]
    )
    return mapping


def classify_raw_type(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    joined = "\\".join(parts)
    if "\\electrocatalysis\\cv\\20\\" in joined:
        return "CV20"
    if "\\electrocatalysis\\eis\\" in joined:
        return "EIS"
    if "\\electrocatalysis\\et\\" in joined:
        return "Et"
    if "\\electrocatalysis\\lsv\\" in joined and path.suffix.lower() == ".txt":
        return "LSV"
    return None


def build_raw_hash_index() -> tuple[dict[str, list[Path]], pd.DataFrame]:
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    rows = []
    raw_files = [path for path in RAW_ROOT.rglob("*.txt") if path.is_file()]
    for index, path in enumerate(raw_files, 1):
        if index % 1000 == 0:
            print(f"Hashed {index}/{len(raw_files)} raw archive files")
        digest = sha256(path)
        data_type = classify_raw_type(path)
        hash_to_paths[digest].append(path)
        rows.append(
            {
                "raw_path": str(path),
                "raw_relpath": rel(path),
                "data_type": data_type,
                "raw_size": path.stat().st_size,
                "sha256": digest,
            }
        )
    return hash_to_paths, pd.DataFrame(rows)


def current_files() -> pd.DataFrame:
    rows = []
    for data_type, spec in DATA_TYPES.items():
        current_dir = spec["current_dir"]
        for path in sorted(current_dir.glob("*.txt")):
            rows.append(
                {
                    "sample": path.stem,
                    "data_type": data_type,
                    "current_path": str(path),
                    "current_relpath": rel(path),
                    "current_size": path.stat().st_size,
                    "current_sha256": sha256(path),
                }
            )
    return pd.DataFrame(rows)


def choose_expected_candidate(current_row: pd.Series, expected: pd.DataFrame) -> pd.Series | None:
    sample = current_row["sample"]
    data_type = current_row["data_type"]
    candidates = expected[
        expected["sample"].eq(sample)
        & expected["data_type"].eq(data_type)
        & expected["expected_raw_exists"]
    ].copy()
    if candidates.empty:
        return None
    candidates["mapping_rank"] = candidates["mapping_file"].map(
        lambda value: mapping_priority(Path(value))
    )
    candidates["candidate_sha256"] = candidates["expected_raw_path"].map(lambda value: sha256(Path(value)))
    candidates["hash_matches_current"] = candidates["candidate_sha256"].eq(
        current_row["current_sha256"]
    )
    candidates = candidates.sort_values(
        ["hash_matches_current", "mapping_rank", "expected_raw_path"],
        ascending=[False, True, True],
    )
    return candidates.iloc[0]


def run_audit() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    expected = build_expected_mapping()
    print(f"Expected mapping rows: {len(expected)}")
    expected.to_csv(OUT_DIR / "expected_raw_mapping_candidates.csv", index=False)

    raw_hash_index, raw_hash_table = build_raw_hash_index()
    raw_hash_table.to_csv(OUT_DIR / "raw_archive_hash_index.csv", index=False)
    print(f"Raw archive hash rows: {len(raw_hash_table)}")

    current = current_files()
    print(f"Current all_data txt files: {len(current)}")

    audit_rows = []
    exact_rows = []
    expected_keyed = expected
    for _, row in current.iterrows():
        exact_paths = raw_hash_index.get(row["current_sha256"], [])
        same_type_exact = [path for path in exact_paths if classify_raw_type(path) == row["data_type"]]
        for raw_path in exact_paths:
            exact_rows.append(
                {
                    "sample": row["sample"],
                    "data_type": row["data_type"],
                    "current_path": row["current_path"],
                    "raw_match_path": str(raw_path),
                    "raw_match_relpath": rel(raw_path),
                    "raw_match_inferred_type": classify_raw_type(raw_path),
                    "raw_match_size": raw_path.stat().st_size,
                    "sha256": row["current_sha256"],
                }
            )

        expected_candidate = choose_expected_candidate(row, expected_keyed)
        if expected_candidate is None:
            expected_path = pd.NA
            expected_relpath = pd.NA
            expected_size = pd.NA
            expected_sha = pd.NA
            expected_matches = False
            status = (
                "no_expected_mapping_exact_match"
                if exact_paths
                else "no_expected_mapping_no_exact_match"
            )
        else:
            expected_path_obj = Path(expected_candidate["expected_raw_path"])
            expected_path = str(expected_path_obj)
            expected_relpath = rel(expected_path_obj)
            expected_size = expected_path_obj.stat().st_size
            expected_sha = sha256(expected_path_obj)
            expected_matches = expected_sha == row["current_sha256"]
            if expected_matches:
                status = "expected_match"
            elif exact_paths:
                status = "expected_mismatch_exact_elsewhere"
            else:
                status = "expected_mismatch_no_exact_elsewhere"

        audit_rows.append(
            {
                "sample": row["sample"],
                "data_type": row["data_type"],
                "status": status,
                "current_path": row["current_path"],
                "current_relpath": row["current_relpath"],
                "current_size": row["current_size"],
                "expected_raw_path": expected_path,
                "expected_raw_relpath": expected_relpath,
                "expected_raw_size": expected_size,
                "expected_hash_matches_current": expected_matches,
                "exact_raw_hash_match_count_full_archive": len(exact_paths),
                "exact_raw_hash_matches_full_archive": " | ".join(rel(path) for path in exact_paths[:20]),
                "exact_raw_hash_match_count_same_type": len(same_type_exact),
                "exact_raw_hash_matches_same_type": " | ".join(rel(path) for path in same_type_exact[:20]),
                "current_sha256": row["current_sha256"],
                "expected_sha256": expected_sha,
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values(["status", "data_type", "sample"])
    exact = pd.DataFrame(exact_rows).sort_values(["data_type", "sample", "raw_match_path"])
    audit.to_csv(OUT_DIR / "all_data_raw_audit.csv", index=False)
    exact.to_csv(OUT_DIR / "all_data_exact_raw_hash_matches.csv", index=False)

    anomalies = audit[audit["status"].ne("expected_match")].copy()
    anomalies.to_csv(OUT_DIR / "all_data_raw_audit_anomalies.csv", index=False)

    current_keys = set(zip(current["sample"], current["data_type"]))
    expected_existing = expected[expected["expected_raw_exists"]].copy()
    expected_existing["current_present"] = [
        (sample, data_type) in current_keys
        for sample, data_type in zip(expected_existing["sample"], expected_existing["data_type"])
    ]
    expected_missing_current = expected_existing[~expected_existing["current_present"]].copy()
    expected_missing_current.to_csv(
        OUT_DIR / "expected_raw_files_missing_from_all_data.csv", index=False
    )

    summary = (
        audit.groupby(["data_type", "status"], as_index=False)
        .size()
        .rename(columns={"size": "n_files"})
        .sort_values(["data_type", "status"])
    )
    summary.to_csv(OUT_DIR / "all_data_raw_audit_summary_by_type.csv", index=False)

    current_counts = current.groupby("data_type", as_index=False).size().rename(
        columns={"size": "n_current_txt_files"}
    )
    expected_counts = (
        expected_existing.groupby("data_type", as_index=False)
        .size()
        .rename(columns={"size": "n_expected_existing_raw_files"})
    )
    counts = current_counts.merge(expected_counts, on="data_type", how="outer").fillna(0)
    counts.to_csv(OUT_DIR / "all_data_raw_audit_counts.csv", index=False)

    lines = [
        "all_data vs raw_data audit",
        "===========================",
        "",
        f"Current all_data txt files: {len(current)}",
        f"Expected raw mapping candidates: {len(expected)}",
        f"Raw archive txt files hashed: {len(raw_hash_table)}",
        "",
        "Counts by data type:",
        counts.to_string(index=False),
        "",
        "Status summary:",
        summary.to_string(index=False),
        "",
        "Anomalies:",
        anomalies[
            [
                "sample",
                "data_type",
                "status",
                "current_relpath",
                "expected_raw_relpath",
                "exact_raw_hash_matches_full_archive",
                "exact_raw_hash_matches_same_type",
            ]
        ].head(200).to_string(index=False),
    ]
    (OUT_DIR / "all_data_raw_audit_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"Audit outputs written to: {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_audit()
