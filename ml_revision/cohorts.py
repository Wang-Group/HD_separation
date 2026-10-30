"""Regenerate cohorts, targets and split memberships from packaged inputs."""
import copy
import numpy as np
import pandas as pd
from descriptors import COVARIATES
from model_core import save
from workflow.data import ROOT, load_data
from workflow.splits import design

KEYS = ['workflow', 'protocol', 'repeat', 'fold', 'arm']


def eta_targets(data):
    folder = ROOT / 'inputs/lsv_crossings'
    curves = pd.read_csv(folder / 'eta10_curves.csv')
    means = pd.read_csv(folder / 'eta10_samples.csv').set_index('sample')
    excluded = pd.read_csv(folder / 'excluded_bo_references.csv')
    assert curves.curve_id.is_unique
    assert not set(curves.curve_id) & set(excluded.curve_id)
    assert curves.loc[curves.status.ne('eta10_crossing'), 'eta10_mV'].isna().all()
    rebuilt = curves.loc[curves.status.eq('eta10_crossing')].groupby('sample').eta10_mV.agg(['mean', 'std', 'size'])
    valid = means.loc[means.n_eta10.gt(0)].sort_index()
    rebuilt = rebuilt.loc[valid.index]
    np.testing.assert_allclose(rebuilt['mean'], valid.eta10_mean_mV, atol=1e-10)
    np.testing.assert_allclose(rebuilt['std'], valid.eta10_SD_mV, atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(rebuilt['size'], valid.n_eta10)
    original = curves.loc[curves.source_relative_path.eq('LSV/' + curves.source_sample + '.txt')].copy()
    for e in pd.read_csv(folder / 'numeric_equivalents.csv').itertuples():
        assert e.sample in ['Ag-Ce', 'Mo-Mg']
        assert e.max_potential_difference == e.max_current_difference == 0
        row = curves.loc[curves['sample'].eq(e.sample) & curves.sha256.eq(e.retained_sha256)]
        assert len(row) == 1 and row.area_cm2.iloc[0] == 2
        original = pd.concat([original, row], ignore_index=True)
    assert original['sample'].is_unique
    original = original.set_index('sample')
    initial = data.loc[data.cohort.eq('initial')]
    audit = initial[['sample']].reset_index(drop=True)
    for c in ['status', 'eta10_mV', 'sha256', 'area_cm2']:
        audit['original_' + c] = audit['sample'].map(original[c])
    audit['averaged_eta10_mV'] = audit['sample'].map(means.eta10_mean_mV)
    audit['n_accepted'] = audit['sample'].map(means.n_eta10)
    audit['experimental_SD_mV'] = audit['sample'].map(means.eta10_SD_mV)
    audit['included'] = audit.original_status.eq('eta10_crossing') & audit.averaged_eta10_mV.notna()
    for arm, column in [('original_only', 'original_eta10_mV'), ('averaged', 'averaged_eta10_mV')]:
        audit[arm + '_target_mV'] = audit[column].clip(10, 550)
    assert audit.included.sum() == 298
    return audit


def memberships(jobs):
    outer, inner = [], []
    for j in jobs:
        key = {k: j[k] for k in KEYS}
        outer.extend(dict(**key, role=role, sample=s, fit_seed=j['seed'])
                     for role in ('train', 'test') for s in j[role]['sample'])
        inner.extend(dict(**key, inner_fold=i, role=role, sample=s, fit_seed=j['seed']+i)
                     for i, (a, b) in enumerate(j['inner'])
                     for role, part in [('train', a), ('validation', b)] for s in part['sample'])
    return pd.DataFrame(outer), pd.DataFrame(inner)


def check_locks(jobs):
    mapping = {'hd_icp': 'hd_icp', 'hd_precursor': 'hd_precursor', 'hd_fraction': 'hd_explicit_fraction',
               'ec_unshifted_measured': 'electrochemical_sensitivity_paired',
               'ec_shifted': 'electrochemical_sensitivity_paired',
               'ec_shifted_measured': 'electrochemical_sensitivity_paired',
               'eta10': 'eta10_target_comparison'}
    for name, current in zip(['outer_memberships', 'inner_memberships'], memberships(jobs)):
        for workflow, part in current.groupby('workflow'):
            locked = pd.read_csv(ROOT / 'inputs/fold_locks' / mapping[workflow] / (name+'.csv'))
            cols = [c for c in part if c in locked and c != 'fit_seed']
            pd.testing.assert_frame_equal(part[cols].sort_values(cols).reset_index(drop=True),
                                          locked[cols].sort_values(cols).reset_index(drop=True))


def prepare(config, output=None):
    data, audit = load_data()
    data['target'] = data.separation_coefficient
    base = design(data, config, include_bo=True)
    jobs = []
    for name, scheme in [('hd_icp', 'icp'), ('hd_precursor', 'precursor'), ('hd_fraction', 'fraction')]:
        jobs += [dict(j, workflow=name, scheme=scheme) for j in base]
    cov = pd.read_csv(ROOT / 'inputs/paired/sample_covariates.csv').set_index('sample')
    assert cov.index.is_unique
    for name, scheme in [('ec_unshifted_measured', 'unshifted_measured'),
                         ('ec_shifted', 'shifted'), ('ec_shifted_measured', 'shifted_measured')]:
        for b in base:
            if b['protocol'] != 'stratified_random' or b['arm'] != 'without_bo':
                continue
            j = copy.deepcopy(b)
            j.update(workflow=name, scheme=scheme)
            for f in [j['train'], j['test'], *[part for pair in j['inner'] for part in pair]]:
                for c in COVARIATES:
                    f[c] = f['sample'].map(cov[c])
                assert np.isfinite(f[COVARIATES].to_numpy()).all()
            jobs.append(j)
    targets = eta_targets(data)
    matched = data.loc[data['sample'].isin(targets.loc[targets.included, 'sample'])].copy()
    table = targets.set_index('sample')
    matched['target'] = matched['sample'].map(table.original_eta10_mV)
    for b in design(matched, config, include_bo=False):
        for arm in ['original_only', 'averaged']:
            j = copy.deepcopy(b)
            j.update(workflow='eta10', scheme='icp', arm=arm)
            for f in [j['train'], j['test'], *[part for pair in j['inner'] for part in pair]]:
                f['target'] = f['sample'].map(table[arm+'_target_mV'])
                assert f.target.between(10, 550).all()
            jobs.append(j)
    assert len(jobs) == 285
    check_locks(jobs)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        targets.to_csv(output / 'eta10_target_audit.csv', index=False)
        audit.to_csv(output / 'cohort_audit.csv', index=False)
        for name, f in zip(['outer_memberships', 'inner_memberships'], memberships(jobs)):
            f.to_csv(output / (name+'.csv'), index=False)
        save(output / 'data_validation.json', dict(initial=560, eligible_bo=111,
             eta10_matched=298, eta10_crossings=int(targets.loc[targets.included, 'n_accepted'].sum()),
             jobs=len(jobs), all_outer_and_inner_memberships_match=True,
             ec_unshifted_source='hd_icp / stratified_random / without_bo',
             bo_references_in_models=False, paired_electrochemical_inputs_unchanged=True))
    return jobs
