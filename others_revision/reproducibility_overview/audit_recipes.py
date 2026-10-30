"""Verify recipe aliases against JSON and raw-curve hashes; no source writes."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RAW = REPO / 'hd_separation_ml/data_extraction/all_data/LSV_repeat'
SERIES = [
    ('Co-Fe', '11-Co-1', 11, '3-CoFe-1', 'initial_sampling_Co_Fe_1.json'),
    ('Co-Mo', '11-Co-1', 8, '1-CoMo', 'initial_sampling_Co_Mo.json'),
    ('Co-Ni', '11-Co-1', 14, '4-CoNi', 'initial_sampling_Co_Ni.json'),
    ('Co-V', '11-Co-1', 6, '2-CoV', 'initial_sampling_Co_V.json'),
    ('Fe-Ni', '13-Fe-1', 14, '5-FeNi', 'initial_sampling_Fe_Ni.json'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe(data, index):
    key = str(index)
    result = {}
    for i in range(1, 5):
        if f'volume{i}' not in data:
            continue
        volume = float(data[f'volume{i}'][key])
        if volume:
            name = data[f'substance{i}'][key]
            result[name] = result.get(name, 0) + volume
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(RAW / 'manifest.csv')
    groups = []
    for sample, initial_folder, index, bo_folder, bo_name in SERIES:
        initial_path = args.source_root / initial_folder / 'initial_sampling.json'
        bo_path = args.source_root / 'BO-raw-data' / bo_folder / 'tmp' / bo_name
        initial = json.loads(initial_path.read_text(encoding='utf-8-sig'))
        bo = json.loads(bo_path.read_text(encoding='utf-8-sig'))
        expected = recipe(initial, index)
        rows = manifest.loc[manifest.sample_group.isin([sample, 'reference_' + sample])]
        assert rows.sha256.is_unique, sample
        reference_rows = rows.loc[rows.sample_group.eq('reference_' + sample)]
        checks = []
        for row in reference_rows.to_dict('records'):
            number = int(Path(row['file']).stem.removeprefix(sample))
            actual = recipe(bo, number)
            assert actual == expected, (sample, number, actual, expected)
            source = bo_path.parent / 'electrocatalysis/LSV/2' / f'{number}.txt'
            assert digest(source) == digest(RAW / row['file']) == row['sha256']
            checks.append(dict(index=number, nonzero_recipe=actual,
                file=row['file'], source=source.relative_to(args.source_root).as_posix(),
                sha256=row['sha256']))
        initial_raw = initial_path.parent / 'tmp/electrocatalysis/LSV' / f'{index}.txt'
        initial_hash = digest(initial_raw)
        groups.append(dict(sample=sample, lsv_alias='reference_' + sample,
            HD_alias=sample if sample == 'Co-Fe' else sample + '3',
            initial=dict(source=initial_path.relative_to(args.source_root).as_posix(),
                sha256=digest(initial_path), index=index, nonzero_recipe=expected,
                LSV_sha256=initial_hash,
                matching_packaged_files=rows.loc[rows.sha256.eq(initial_hash), 'file'].tolist()),
            BO_reference=dict(source=bo_path.relative_to(args.source_root).as_posix(),
                sha256=digest(bo_path), nonzero_recipe=expected, checked_curves=checks),
            n_packaged_LSV=len(rows), n_unique_packaged_LSV=rows.sha256.nunique()))
    target = HERE / 'recipe_evidence.json'
    evidence = json.loads(target.read_text(encoding='utf-8'))
    evidence['groups'] = groups
    evidence['lsv_aliases'] = {r['lsv_alias']: r['sample'] for r in groups}
    evidence['HD_aliases'] = {r['HD_alias']: r['sample'] for r in groups if r['HD_alias'] != r['sample']}
    evidence['decision'] = 'Pool measurements with matching recorded precursor recipes under ordinary catalyst names. Each retained measurement is counted once.'
    evidence['limits'] = ('Recipe matching verifies recorded substance names and volumes, not stock concentrations or identical acquisition conditions. '
        'The current Co-V LSV export does not byte-match initial batch 11-Co-1/index 6; its existing manifest identity and area are retained. '
        'No raw curves are added. JSON processed electrochemical/H-D values are not used. '
        'Only adopted H/D measurements are pooled; no omitted BO measurement is reinstated.')
    target.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(lsv_aliases=evidence['lsv_aliases'], HD_aliases=evidence['HD_aliases']), indent=2))


if __name__ == '__main__':
    main()
