"""Rebuild intrinsic inputs from the packaged ICP workbooks and reference tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASES = ['work_function', 'pzc', 'MM_bond', 'E_std', 'FE_ads_avg', 'delta_EOH',
         'electronegativity', 'MH_bond_kJ', 'Frequency_OH', 'Frequency_H']
RAW = [f'{base}_{side}' for base in BASES for side in ('element1', 'element2')]
FEATURES = RAW + ['icp_weighted_' + base for base in BASES]
EXCLUDED_BASES = ('MD_bond_kJ', 'Frequency_D', 'delta_ZPE_H')
assert len(BASES) == 10 and len(FEATURES) == 30


def fractions(a, b, ma, mb):
    if not np.isfinite([a, b, ma, mb]).all() or min(ma, mb) <= 0:
        return [np.nan] * 4
    a, b = max(a, 0), max(b, 0)
    if a + b <= 0:
        return [np.nan] * 4
    f = (a / ma) / (a / ma + b / mb)
    return [f, 1-f, a/(a+b), b/(a+b)]


def load_data():
    folder = ROOT / 'inputs'
    frame = pd.read_excel(folder / 'icp/ICP_raw_and_percentages_EN.xlsx', sheet_name='ICP')
    assert frame.Sample.is_unique and frame.Sample.notna().all()
    masses = {}
    records = []
    for _, r in frame.iterrows():
        for side in (1, 2):
            metal, mass = r[f'M{side}'], r[f'M{side} atomic mass']
            if metal in masses and np.isfinite(masses[metal]) and np.isfinite(mass):
                np.testing.assert_allclose(masses[metal], mass)
            if metal not in masses or np.isfinite(mass):
                masses[metal] = mass
        expected = [1., 0., 1., 0.] if r.M1 == r.M2 else fractions(
            r['M1 raw value'], r['M2 raw value'], masses[r.M1], masses[r.M2])
        actual = [r[f'M{s} {kind} percentage'] for kind in ('molar', 'mass') for s in (1, 2)]
        np.testing.assert_allclose(expected, actual, atol=1e-10, rtol=0, equal_nan=True)
        shared = '' if pd.isna(r['Shared ICP ID']) else str(r['Shared ICP ID'])
        records.append(dict(sample=r.Sample, element1=r.M1, element2=r.M2,
            cohort='initial' if r.Dataset == 'Initial screening' else 'BO',
            actual_frac_element1=expected[0], actual_frac_element2=expected[1],
            icp_group='shared:'+shared if shared else 'sample:'+r.Sample))
    data = pd.DataFrame(records)
    labels = pd.read_csv(folder / 'reference/original_targets.csv')
    assert labels['sample'].is_unique
    data = data.merge(labels[['sample','separation_coefficient']], on='sample', how='left', validate='one_to_one')
    data['status'] = np.where(data.actual_frac_element1.isna(), 'no_valid_icp_composition',
                      np.where(data.separation_coefficient.isna(), 'missing_target', 'eligible'))
    representatives = data.assign(priority=data.cohort.ne('initial')).sort_values('priority', kind='stable').drop_duplicates('icp_group')['sample']
    data.loc[~data['sample'].isin(representatives), 'status'] = 'shared_icp_not_independent'
    raw = pd.read_excel(folder / 'icp/CoFe_ICP-OES_received.xlsx', sheet_name='Sheet1')
    ids = raw.iloc[:, 0].astype(str).str.extract(r'^CoFe-(\d+)$')[0]
    raw = raw.loc[ids.notna()].copy()
    raw.index = 'Co-Fe' + ids.loc[raw.index]
    assert raw.index.is_unique
    cofe = []
    for r in pd.read_csv(folder / 'reference/CoFe_targets.csv').itertuples():
        f = [np.nan]*4
        status = 'no_individual_icp'
        if r.sample in raw.index:
            a, b = raw.loc[r.sample].iloc[1:3].to_numpy(float)
            f = fractions(a, b, masses['Co'], masses['Fe'])
            status = 'eligible' if np.isfinite(f[0]) else 'no_valid_icp_composition'
        if r.sample == 'Co-Fe3':
            np.testing.assert_allclose(f[0], data.loc[data['sample'].eq('Co-Fe'), 'actual_frac_element1'].iloc[0], atol=1e-12)
            status = 'shared_icp_not_independent'
        if not np.isfinite(r.separation_coefficient) and status == 'eligible':
            status = 'missing_target'
        cofe.append(dict(sample=r.sample, element1='Co', element2='Fe', cohort='BO',
            actual_frac_element1=f[0], actual_frac_element2=f[1], status=status,
            separation_coefficient=r.separation_coefficient, icp_group='sample:'+r.sample))
    audit = pd.concat([data, pd.DataFrame(cofe)], ignore_index=True)
    assert audit['sample'].is_unique
    data = audit.loc[audit.status.eq('eligible')].copy().set_index('sample', drop=False)
    properties = pd.read_csv(folder / 'reference/elemental_properties.csv').set_index('element')
    assert properties.index.is_unique
    for side in ('element1','element2'):
        for base in BASES:
            data[f'{base}_{side}'] = data[side].map(properties[base])
    meta = pd.read_csv(folder / 'reference/sample_metadata.csv').set_index('sample')
    data['nominal_frac_element1'] = data['sample'].map(meta.nominal_frac_element1)
    four = pd.read_csv(folder / 'reference/BO_recipes.csv')
    four['sample'] = four.system + four.row_id.astype(str)
    cf = pd.DataFrame(json.loads((folder / 'reference/CoFe_recipes.json').read_text(encoding='utf-8')))
    cf['sample'] = 'Co-Fe' + cf.index.astype(str)
    recipes = pd.concat([four, cf]).set_index('sample', verify_integrity=True)
    for sample, r in data.loc[data.cohort.eq('BO')].iterrows():
        recipe = recipes.loc[sample]
        assert (r.element1, r.element2) == (recipe.substance1, recipe.substance2)
        np.testing.assert_allclose(recipe.separation_coefficient, r.separation_coefficient, atol=1e-7, rtol=0)
        assert min(recipe.volume1, recipe.volume2) >= 0 and recipe.volume1 + recipe.volume2 > 0
        data.loc[sample, 'nominal_frac_element1'] = recipe.volume1 / (recipe.volume1 + recipe.volume2)
    data['metal_pair'] = data[['element1','element2']].apply(lambda r: '-'.join(sorted(r)), axis=1)
    first = data.metal_pair.str.split('-').str[0]
    frac = data.actual_frac_element1.where(data.element1.eq(first), 1-data.actual_frac_element1)
    bins = np.minimum(np.floor((frac + 1e-12)/.05).astype(int), 19)
    data['composition_group'] = data.metal_pair + '_bin' + bins.astype(str)
    assert data.nominal_frac_element1.between(0,1).all()
    assert np.isfinite(data.separation_coefficient).all()
    assert data.cohort.value_counts().to_dict() == {'initial':560, 'BO':111}
    return data, audit


class MaterialDescriptors:
    def __init__(self, weighting='icp'):
        if weighting not in ('icp','precursor'):
            raise ValueError(weighting)
        self.weighting = weighting

    def fit(self, data):
        self.medians = {b: float(np.nanmedian(data[[b+'_element1', b+'_element2']].to_numpy(float))) for b in BASES}
        if not np.isfinite(list(self.medians.values())).all():
            raise ValueError('Property entirely missing in training fold')
        return self

    def transform(self, data):
        out = pd.DataFrame({c:data[c].fillna(self.medians[c.rsplit('_',1)[0]]).to_numpy(float) for c in RAW}, index=data.index)
        f = data['actual_frac_element1' if self.weighting == 'icp' else 'nominal_frac_element1'].to_numpy(float)
        if not np.isfinite(f).all() or ((f < 0)|(f > 1)).any():
            raise ValueError('Invalid composition')
        for b in BASES:
            out['icp_weighted_'+b] = f*out[b+'_element1'] + (1-f)*out[b+'_element2']
        return out[FEATURES]
