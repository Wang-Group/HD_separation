"""One curve-level aggregation for Figure 1B and repeatability plots.

Extraction records remain untouched. Pool only the audited same-recipe
labels; keep crossings, estimates, and bounds in separate numerical fields.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RECIPE = REPO / 'others_revision/reproducibility_overview/recipe_evidence.json'
ALIASES = json.loads(RECIPE.read_text(encoding='utf-8'))['lsv_aliases']
ESTIMATE_STATUSES = ['rising_point', 'nearest_point_estimate']


def inputs():
    return [HERE / name for name in ['eta10_curves.csv', 'eta10_samples.csv',
            'extraction_summary.json', 'lsv_rules.py', 'lsv_noise_exceptions.json']] + [RECIPE]


def input_hashes():
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs() + [Path(__file__)]}


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return (float(np.mean(values)) if len(values) else np.nan,
            float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan)


def aggregate(curves):
    rows = []
    for sample, group in curves.groupby('sample', sort=True):
        row = dict(sample=sample, raw_curves=len(group),
                   source_samples=';'.join(sorted(group.source_sample.unique())))
        specs = [('eta10_crossing', 'eta10', 'eta10_mV'),
                 ('rising_point', 'rising_point', 'estimate_potential_mV'),
                 ('nearest_point_estimate', 'nearest_point', 'estimate_potential_mV')]
        for status, prefix, field in specs:
            subset = group.loc[group.status.eq(status)]
            row[prefix + '_mean_mV'], row[prefix + '_SD_mV'] = mean_sd(subset[field])
            count = {'eta10': 'n_eta10', 'rising_point': 'n_rising_points',
                     'nearest_point': 'n_nearest_point_estimates'}[prefix]
            row[count] = len(subset)
        estimates = group.loc[group.status.isin(ESTIMATE_STATUSES)]
        row['n_estimates'] = len(estimates)
        row['estimate_mean_mV'], row['estimate_SD_mV'] = mean_sd(estimates.estimate_potential_mV)
        row['estimate_j_mean_mA_cm2'], row['estimate_j_SD_mA_cm2'] = mean_sd(estimates.estimate_j_mA_cm2)
        row['estimate_types'] = ';'.join(sorted(estimates.status.unique()))
        row['n_scan_limit'] = int(group.status.eq('scan_limit').sum())
        row['scan_limit_mV'] = group.scan_limit_mV.max()
        row['n_above_target_no_crossing'] = int(group.status.eq('above_target_no_crossing').sum())
        rows.append(row)
    return pd.DataFrame(rows)


def load_data():
    evidence = json.loads(RECIPE.read_text(encoding='utf-8'))
    for group in evidence['groups']:
        assert group['initial']['nonzero_recipe'] == group['BO_reference']['nonzero_recipe']
    curves = pd.read_csv(HERE / 'eta10_curves.csv')
    assert curves.curve_id.is_unique
    assert curves.loc[curves.status.ne('eta10_crossing'), 'eta10_mV'].isna().all()
    assert curves.loc[~curves.status.isin(ESTIMATE_STATUSES), 'estimate_potential_mV'].isna().all()
    curves['source_sample'] = curves['sample']
    unpooled = aggregate(curves).set_index('sample')
    saved = pd.read_csv(HERE / 'eta10_samples.csv').set_index('sample').loc[unpooled.index]
    for column in ['raw_curves', 'n_eta10', 'eta10_mean_mV', 'eta10_SD_mV',
                   'n_rising_points', 'n_nearest_point_estimates', 'n_estimates',
                   'n_scan_limit', 'n_above_target_no_crossing']:
        np.testing.assert_allclose(unpooled[column], saved[column], atol=1e-9, equal_nan=True)
    curves['sample'] = curves['source_sample'].replace(ALIASES)
    for group in evidence['groups']:
        pooled = curves.loc[curves['sample'].eq(group['sample'])]
        assert len(pooled) == pooled.sha256.nunique() == group['n_packaged_LSV']
    stats = aggregate(curves)
    assert len(stats) == len(saved) - len(ALIASES)
    stats['potential_metric'] = np.select(
        [stats.n_eta10.gt(0), stats.n_estimates.gt(0), stats.n_scan_limit.gt(0)],
        ['eta10', 'estimate', 'scan_limit'], default='no_accepted_result')
    stats['potential_mean_mV'] = np.select(
        [stats.potential_metric.eq('eta10'), stats.potential_metric.eq('estimate'),
         stats.potential_metric.eq('scan_limit')],
        [stats.eta10_mean_mV, stats.estimate_mean_mV, stats.scan_limit_mV], default=np.nan)
    stats['potential_SD_mV'] = np.select(
        [stats.potential_metric.eq('eta10'), stats.potential_metric.eq('estimate')],
        [stats.eta10_SD_mV, stats.estimate_SD_mV], default=np.nan)
    stats['n_potential'] = np.select(
        [stats.potential_metric.eq('eta10'), stats.potential_metric.eq('estimate')],
        [stats.n_eta10, stats.n_estimates], default=0)
    return curves, stats


def original_hd():
    data = pd.read_csv(HERE / 'figure1b_inputs/original_HD.csv')
    for alias, sample in ALIASES.items():
        source = data.loc[data['sample'].isin([sample, alias])]
        assert len(source) == 2
        row = source.loc[source['sample'].eq(sample)].copy()
        row['original_HD_coefficient'] = np.average(source.original_HD_coefficient,
                                                  weights=source.original_HD_count)
        row['original_HD_count'] = int(source.original_HD_count.sum())
        row['catalyst_class'] = 'same_recipe_measurements'
        data = pd.concat([data.loc[~data['sample'].isin(source['sample'])], row], ignore_index=True)
    assert data['sample'].is_unique
    return data
