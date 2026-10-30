"""Assign representative ICP metadata to audited BO references without fitting."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from workflow.data import ROOT, load_data
from workflow.recipes import packaged_recipes, same_full_recipe

FOLDER = ROOT / 'reference_icp'
BOOK = ROOT / 'inputs/icp/ICP_raw_and_percentages_EN.xlsx'
INPUTS = ['reference_icp/recipe_groups.json', 'inputs/icp/ICP_raw_and_percentages_EN.xlsx',
          'inputs/icp/CoFe_ICP-OES_received.xlsx', 'inputs/reference/BO_recipes.csv',
          'inputs/reference/CoFe_recipes.json', 'reference_icp.py']


def build_assignments(frame=None, audit=None, recipes=None, groups=None):
    frame = pd.read_excel(BOOK, sheet_name='ICP') if frame is None else frame.copy()
    if audit is None:
        _, audit = load_data()
    if recipes is None:
        recipes = packaged_recipes()
    if groups is None:
        groups = json.loads((FOLDER / 'recipe_groups.json').read_text(encoding='utf-8'))['groups']
    assert frame.Sample.is_unique and audit['sample'].is_unique
    excel_rows = dict(zip(frame.Sample, range(2, len(frame) + 2)))
    frame = frame.set_index('Sample')
    audit = audit.set_index('sample')
    rows = []
    for group in groups:
        initial, source = group['sample'], group['source_icp_id']
        recipe = tuple(sorted(group['recipe'].items()))
        matches = sorted(sample for sample, candidate in recipes.items()
                         if same_full_recipe(recipe, candidate))
        assert len(matches) == group['expected_bo_references'], (initial, matches)
        assert source in matches
        donor = frame.loc[initial]
        assert donor['Shared ICP ID'] == source
        metals = [donor.M1, donor.M2]
        assert all(m in group['recipe'] for m in metals)
        fractions = donor[['M1 molar percentage', 'M2 molar percentage']].to_numpy(float)
        assert np.isfinite(fractions).all() and (fractions >= 0).all()
        np.testing.assert_allclose(fractions.sum(), 1., rtol=0, atol=1e-12)
        for sample in [initial] + matches:
            entry = audit.loc[sample]
            assert [entry.element1, entry.element2] == metals
            existing = entry[['actual_frac_element1', 'actual_frac_element2']].to_numpy(float)
            if np.isfinite(existing).any():
                np.testing.assert_allclose(existing, fractions, rtol=0, atol=1e-12,
                                           err_msg=f'Conflicting ICP composition: {sample}')
            else:
                assert entry.status == 'no_individual_icp', sample
            if sample in frame.index:
                assert frame.loc[sample, 'Shared ICP ID'] == source, sample
            assert entry.status == ('eligible' if sample == initial else
                                    'no_individual_icp' if not np.isfinite(existing).any() else
                                    'shared_icp_not_independent'), sample
            rows.append(dict(sample=sample, series=initial, cohort=entry.cohort,
                recipe_group='recipe:' + initial, icp_group='shared:' + source,
                source_icp_id=source,
                assignment='measured_source' if sample == source else 'shared_recipe_reference',
                source_is_this_sample=sample == source,
                individual_icp_record_available=sample == source,
                representative_for_model=initial, included_in_current_ml=sample == initial,
                mapping_completed_missing_composition=not np.isfinite(existing).any(),
                element1=metals[0], element2=metals[1],
                actual_frac_element1=fractions[0], actual_frac_element2=fractions[1],
                volume1=group['recipe'][metals[0]], volume2=group['recipe'][metals[1]],
                citrate_volume=group['recipe']['Na3Cit'],
                nominal_frac_element1=group['recipe'][metals[0]] /
                    (group['recipe'][metals[0]] + group['recipe'][metals[1]]),
                raw_value1=donor['M1 raw value'], raw_value2=donor['M2 raw value'],
                raw_unit=donor['Raw value unit'],
                icp_workbook=BOOK.relative_to(ROOT).as_posix(), icp_sheet='ICP',
                icp_excel_row=excel_rows[initial],
                icp_source_record=donor['Source record (file type / sample / row)'],
                initial_recipe_source=group['initial_source'],
                initial_recipe_sha256=group['source_sha256'],
                initial_recipe_index=group['initial_index'],
                bo_recipe_file='inputs/reference/CoFe_recipes.json' if initial == 'Co-Fe'
                    else 'inputs/reference/BO_recipes.csv'))
    result = pd.DataFrame(rows)
    assert result['sample'].is_unique
    assert set(frame.loc[frame['Shared ICP ID'].notna()].index) <= set(result['sample'])
    return result


def annotate_metadata(audit, assignments):
    """Enrich record metadata; do not admit additional rows to the ML cohort."""
    result = audit.copy().set_index('sample', drop=False)
    for row in assignments.itertuples():
        result.loc[row.sample, ['actual_frac_element1', 'actual_frac_element2']] = [
            row.actual_frac_element1, row.actual_frac_element2]
        for field in ['recipe_group', 'icp_group', 'source_icp_id', 'assignment']:
            result.loc[row.sample, field] = getattr(row, field)
        if not row.included_in_current_ml:
            result.loc[row.sample, 'status'] = 'shared_reference_represented_by_initial'
    pd.testing.assert_frame_equal(audit.loc[audit.status.eq('eligible')].reset_index(drop=True),
        result.loc[result.status.eq('eligible'), audit.columns].reset_index(drop=True))
    return result.reset_index(drop=True)


def load_reference_metadata():
    """Return the unchanged fitting matrix plus consistently assigned metadata."""
    data, audit = load_data()
    return data, annotate_metadata(audit, build_assignments(audit=audit))


def verify_assignments():
    expected = build_assignments()
    saved = pd.read_csv(FOLDER / 'assignments.csv')
    pd.testing.assert_frame_equal(saved, expected, check_exact=False, rtol=1e-14, atol=1e-14)
    data, audit = load_data()
    annotated = annotate_metadata(audit, expected)
    assert not annotated.status.eq('no_individual_icp').any()
    pd.testing.assert_frame_equal(
        audit.loc[~audit['sample'].isin(expected['sample'])].reset_index(drop=True),
        annotated.loc[~annotated['sample'].isin(expected['sample']), audit.columns].reset_index(drop=True))
    return dict(passed=True, model_fits_run=0, series=expected.series.nunique(),
        reference_bo_records=int(expected.cohort.eq('BO').sum()),
        initial_records=int(expected.cohort.eq('initial').sum()),
        unique_icp_sources=expected.source_icp_id.nunique(),
        completed_assignments=int(expected.mapping_completed_missing_composition.sum()),
        cohort_initial=int(data.cohort.eq('initial').sum()), cohort_bo=int(data.cohort.eq('BO').sum()),
        fitting_matrix_unchanged=True, independent_measurements_added=0,
        hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUTS})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify without writing any files.')
    args = parser.parse_args(argv)
    if not args.check:
        build_assignments().to_csv(FOLDER / 'assignments.csv', index=False)
    report = verify_assignments()
    if not args.check:
        (FOLDER / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
