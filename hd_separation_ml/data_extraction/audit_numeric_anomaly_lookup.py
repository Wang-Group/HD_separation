#!/usr/bin/env python3
"""Find numeric raw-data correspondences for hash-audit anomalies.

Some 168 raw_data files contain CSStudio metadata and wide instrument columns,
while all_data stores only extracted numeric columns. This script ignores
headers/metadata and compares numeric matrices for files that the exact SHA256
audit could not map.

Run audit_all_data_against_raw.py first.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path(
    r"C:\Users\yibinjiang\Desktop\corrected_data_source\0-data-archive\raw_data"
)
OUT_DIR = ROOT / "hd_separation_ml" / "data_extraction" / "all_data_raw_audit"
AUDIT_CSV = OUT_DIR / "all_data_raw_audit.csv"
OUT_CSV = OUT_DIR / "all_data_anomalies_numeric_raw_lookup.csv"

NUMERIC_SPLIT_RE = re.compile(r"[\t, ]+")
RTOL = 1e-5
ATOL = 1e-8


def parse_numeric(path: Path, cache: dict[str, np.ndarray | None]) -> np.ndarray | None:
    key = str(path)
    if key in cache:
        return cache[key]

    rows: list[list[float]] = []
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        cache[key] = None
        return None

    for line in text.splitlines():
        parts = [part for part in NUMERIC_SPLIT_RE.split(line.strip()) if part]
        values = []
        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                values = []
                break
        if len(values) >= 2:
            rows.append(values)

    if not rows:
        cache[key] = None
        return None

    counts: dict[int, int] = {}
    for row in rows:
        counts[len(row)] = counts.get(len(row), 0) + 1
    modal_columns = max(counts, key=counts.get)
    matrix = np.array([row for row in rows if len(row) == modal_columns], dtype=float)
    cache[key] = matrix
    return matrix


def candidate_files(data_type: str) -> list[Path]:
    candidates = []
    for path in RAW_ROOT.rglob("*.txt"):
        text_path = str(path).lower()
        name = path.name.lower()
        if data_type == "CV20" and (
            r"electrocatalysis\cv\20" in text_path or "cv20" in name
        ):
            candidates.append(path)
        elif data_type == "EIS" and (
            r"electrocatalysis\eis" in text_path or "eis" in name
        ):
            candidates.append(path)
        elif data_type == "LSV" and (
            r"electrocatalysis\lsv" in text_path or "lsv" in name
        ):
            candidates.append(path)
        elif data_type == "Et" and (
            r"electrocatalysis\et" in text_path
            or "catalysis" in name
            or "galstatic" in name
        ):
            candidates.append(path)
    return candidates


def transformed_candidates(
    raw_matrix: np.ndarray | None, data_type: str, current_columns: int
) -> list[tuple[str, np.ndarray]]:
    if raw_matrix is None:
        return []

    transformed = []
    if raw_matrix.shape[1] == current_columns:
        transformed.append(("same_cols", raw_matrix))

    # Instrument EIS exports are often:
    # Freq, Ampl, Bias, Time, Z', Z'', |Z|, Phase, Range
    # all_data keeps:
    # Freq, Z', Z'', |Z|, Phase
    if data_type == "EIS" and current_columns == 5 and raw_matrix.shape[1] >= 8:
        transformed.append(("eis_cols_0_4_5_6_7", raw_matrix[:, [0, 4, 5, 6, 7]]))

    return transformed


def allclose(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, rtol=RTOL, atol=ATOL, equal_nan=False))


def row_close_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    tolerance = ATOL + RTOL * np.abs(right)
    return np.all(np.abs(left - right) <= tolerance, axis=1)


def one_row_relation(current: np.ndarray, raw_matrix: np.ndarray) -> str | None:
    if current.shape[1] != raw_matrix.shape[1]:
        return None

    if len(current) == len(raw_matrix) + 1:
        prefix_mask = row_close_mask(current[: len(raw_matrix)], raw_matrix)
        mismatch = np.flatnonzero(~prefix_mask)
        delete_index = int(mismatch[0]) if len(mismatch) else len(raw_matrix)
        if allclose(current[delete_index + 1 :], raw_matrix[delete_index:]):
            return f"current_has_one_extra_row_at_index_{delete_index}"

    if len(raw_matrix) == len(current) + 1:
        prefix_mask = row_close_mask(current, raw_matrix[: len(current)])
        mismatch = np.flatnonzero(~prefix_mask)
        delete_index = int(mismatch[0]) if len(mismatch) else len(current)
        if allclose(current[delete_index:], raw_matrix[delete_index + 1 :]):
            return f"raw_has_one_extra_row_at_index_{delete_index}"

    return None


def main() -> None:
    audit = pd.read_csv(AUDIT_CSV)
    targets = audit[
        audit["status"].ne("expected_match")
        & audit["exact_raw_hash_match_count_full_archive"].fillna(0).eq(0)
    ].copy()

    cache: dict[str, np.ndarray | None] = {}
    candidates_by_type = {
        data_type: candidate_files(data_type) for data_type in ["CV20", "EIS", "Et", "LSV"]
    }
    rows = []

    for _, target in targets.iterrows():
        current = parse_numeric(Path(target["current_path"]), cache)
        sample = target["sample"]
        data_type = target["data_type"]
        hits = []

        if current is not None:
            sample_key = sample.lower()
            candidates = sorted(
                candidates_by_type[data_type],
                key=lambda path: (sample_key not in str(path).lower(), str(path)),
            )
            for raw_path in candidates:
                raw_matrix = parse_numeric(raw_path, cache)
                for mode, matrix in transformed_candidates(
                    raw_matrix, data_type, current.shape[1]
                ):
                    if matrix.shape == current.shape and allclose(current, matrix):
                        hits.append((raw_path, mode, "numeric_exact"))
                        break

                    relation = one_row_relation(current, matrix)
                    if relation:
                        hits.append((raw_path, mode, relation))
                        break

                if len(hits) >= 10:
                    break

        rows.append(
            {
                "sample": sample,
                "data_type": data_type,
                "status": target["status"],
                "current_relpath": target["current_relpath"],
                "numeric_match_count": len(hits),
                "numeric_raw_matches": " | ".join(
                    f"{path.relative_to(RAW_ROOT)} [{mode}; {relation}]"
                    for path, mode, relation in hits[:10]
                ),
            }
        )

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
