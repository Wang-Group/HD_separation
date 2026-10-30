"""Rebuild acquisition-paired covariates from packaged raw LSV, EIS and Et."""
import hashlib
import json
import re
import numpy as np
import pandas as pd
from workflow.data import ROOT, load_data


def numeric(path):
    rows = []
    for line in path.read_text(encoding='utf-8-sig', errors='replace').splitlines():
        try:
            row = [float(x) for x in re.split(r'[,\s]+', line.strip()) if x]
        except ValueError:
            continue
        if len(row) >= 3:
            rows.append(row)
    if not rows:
        raise ValueError(str(path))
    return rows


def current_at(potential, current):
    assert np.isfinite(potential).all() and np.isfinite(current).all()
    if not min(potential) <= -1 <= max(potential):
        return np.nan
    x, y = potential[::-1], current[::-1]
    if np.any(np.diff(x) < 0):
        order = np.argsort(potential)
        x, y = potential[order], current[order]
    grid = np.arange(-.92, -1.51, -.02)
    return float(np.interp(grid, x, y)[np.argmin(np.abs(grid + 1))])


def main():
    source = ROOT / 'inputs/electrochemical'
    out = ROOT / 'results/electrochemical_audit'
    out.mkdir(parents=True, exist_ok=True)
    data, _ = load_data()
    initial = data.loc[data.cohort.eq('initial')]
    areas = pd.read_csv(source / 'lsv_geometry_area_metadata.csv').set_index('catalyst').geometry_area_cm2
    inventory = pd.read_csv(source / 'source_inventory.csv')
    for row in inventory.itertuples():
        assert hashlib.sha256((ROOT / row.snapshot).read_bytes()).hexdigest() == row.sha256
    records = []
    for sample in initial.index:
        paths = {k: source / 'instrument_exports' / k / (sample + '.txt') for k in ['LSV', 'EIS', 'Et']}
        assert all(p.is_file() for p in paths.values()), sample
        et = np.asarray(numeric(paths['Et']))[:, :3]
        window = et[(et[:, 2] >= 1100) & (et[:, 2] <= 1200)]
        eis = []
        for r in numeric(paths['EIS']):
            if len(r) >= 9:
                eis.append([r[0], r[4]])
            elif len(r) >= 5:
                eis.append([r[0], r[1]])
        eis = np.asarray(eis)
        eis = eis[np.isfinite(eis).all(axis=1) & (eis[:, 0] > 0)]
        eis = eis[np.argsort(-eis[:, 0], kind='stable')]
        rs = float(np.median(eis[:3, 1]))
        potential = float(np.mean(window[:, 0])) if len(window) else np.nan
        curve = np.asarray(numeric(paths['LSV']))[:, :3]
        area = float(areas.get(sample, 1))
        current = current_at(curve[:, 0], curve[:, 1]) * 1000 / area
        values = dict(E_actual=potential, Rs_from_EIS=rs, I_raw=current,
                      E_IR_corrected_final=potential + .9 * .05 * rs)
        assert np.isfinite(list(values.values())).all(), sample
        records.append(dict(sample=sample, status='complete', area_cm2=area,
            area_basis='explicit_metadata' if sample in areas.index else '1cm2_assumption',
            et_window_n=len(window), et_current_mean_A=float(np.mean(window[:, 1])),
            potential_offset=values['E_IR_corrected_final'] + .098, **values))
    frame = pd.DataFrame(records)
    assert len(frame) == 560 and frame['sample'].is_unique
    frame.to_csv(out / 'sample_covariates.csv', index=False)
    report = dict(complete=True, initial_samples=len(frame), raw_snapshot_hash_checks=len(inventory),
                  current_unit='mA/cm2', current_source='Acquisition-paired LSV only',
                  potential_shift='E_IR_corrected_final + 0.098 V', model_results_read=False)
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
