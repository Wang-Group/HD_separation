"""Review adopted H/D repeat measurements; never modify sources or ML targets."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics as stats

HERE = Path(__file__).resolve().parent
REFERENCE_GROUPS = {'Co-Fe', 'Co-Mo', 'Co-Ni', 'Co-V', 'Fe-Ni'}
SINGLE_METALS = {'Al-Al', 'Ba-Ba', 'Ca-Ca', 'Mg-Mg', 'Pb-Pb', 'Sr-Sr'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    match = re.fullmatch(r'([A-Z][a-z]?)([A-Z][a-z]?)(\d*)', re.sub(r'[\s_-]', '', value))
    if not match:
        raise ValueError(value)
    return f'{match[1]}-{match[2]}{match[3]}'


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    left = int(position)
    fraction = position - left
    return values[left] + fraction * (values[min(left + 1, len(values) - 1)] - values[left])


def distribution(rows):
    sd = [row['SD'] for row in rows]
    cv = [row['CV_percent'] for row in rows]
    return dict(groups=len(rows), median_SD=stats.median(sd), mean_SD=stats.mean(sd),
                Q1_SD=quantile(sd, .25), Q3_SD=quantile(sd, .75), maximum_SD=max(sd),
                median_CV_percent=stats.median(cv), mean_CV_percent=stats.mean(cv),
                Q1_CV_percent=quantile(cv, .25), Q3_CV_percent=quantile(cv, .75),
                SD_at_most_0_5=sum(v <= .5 for v in sd),
                SD_at_most_1=sum(v <= 1 for v in sd),
                CV_at_most_10_percent=sum(v <= 10 for v in cv))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-sources', action='store_true',
                        help='Read available Z: raw files and compare hashes; no source writes.')
    args = parser.parse_args()
    out = HERE / 'tables'
    out.mkdir(exist_ok=True)
    package = json.loads((HERE / 'source_manifest.json').read_text(encoding='utf-8-sig'))
    for item in package['snapshots']:
        assert sha(HERE / item['local']) == item['sha256'], item['local']
    raw_manifest = json.loads((HERE / 'data/raw_source_manifest.json').read_text(encoding='utf-8-sig'))
    raw_sources = {item['snapshot']: item for item in raw_manifest}
    curves = read_csv(HERE / 'data/adopted_coefficients.csv')
    initial = {canonical(row['sample']): row for row in read_csv(HERE / 'data/initial_submission.csv')}
    manifest_rows = read_csv(HERE / 'data/measurement_manifest.csv')
    measurements = {row['record_id']: row for row in manifest_rows}
    assert len(measurements) == len(manifest_rows) == 130
    injections = defaultdict(list)
    for row in read_csv(HERE / 'data/injection_results.csv'):
        injections[row['record_id']].append(row)
    confirmed = set(json.loads((HERE / 'data/independence_confirmation.json').read_text(encoding='utf-8-sig'))['samples'])
    assert len(curves) == 800
    records, injection_summary = [], []
    coexist = set()
    largest_rounding_difference = 0
    for row in curves:
        sample = canonical(row['sample'])
        rid = row['raw_record_id']
        value = float(row['current_S_HD'])
        if rid:
            record = measurements[rid]
            assert record['adopted'] == 'True' and canonical(record['sample']) == sample
            selected = [r for r in injections[rid] if r['used'] == 'True']
            assert sorted(int(r['injection']) for r in selected) == [2, 3]
            used_values = [float(r['S_HD']) for r in selected]
            injection_mean = stats.mean(used_values)
            assert abs(injection_mean - float(record['S_HD'])) < 1e-9
            difference = abs(value - injection_mean)
            assert difference <= 1.5e-5
            largest_rounding_difference = max(largest_rounding_difference, difference)
            source = raw_sources[record['source']]['source']
            metadata = raw_sources[record['metadata']]['source']
            injection_summary.append(dict(record_id=rid, sample=sample,
                coefficient_from_last_two=injection_mean, used_injections=2,
                technical_injection_SD=stats.stdev(used_values),
                technical_injection_CV_percent=100 * stats.stdev(used_values) / injection_mean,
                difference_from_adopted_value=difference))
            if row['category'] == 'historical_reprocessed' and row['initial_value_retained'] == 'True':
                assert sample in confirmed
                coexist.add(sample)
            measurement_id = rid
        else:
            old_id = row['reference_initial_ids'] or sample
            old_id = canonical(old_id)
            assert abs(value - float(initial[old_id]['separation_coefficient'])) <= 1.5e-5
            measurement_id = 'initial:' + old_id
            source = 'Initial-submission coefficient table; identity: ' + old_id
            metadata = ''
        records.append(dict(sample=sample, measurement_id=measurement_id, S_HD=value,
            category=row['category'], reference_group=sample in REFERENCE_GROUPS,
            summary_cell=row['cell'], raw_source=source, identity_metadata=metadata,
            raw_group_1based=measurements[rid]['group_1based'] if rid else '',
            initial_value_retained=row['initial_value_retained']))
    assert coexist == confirmed
    assert len(injection_summary) == 104
    assert len({row['measurement_id'] for row in records}) == 800
    for sample in sorted(SINGLE_METALS):
        assert not any(row['sample'] == sample for row in records)
        records.append(dict(sample=sample, measurement_id='initial:' + sample,
            S_HD=float(initial[sample]['separation_coefficient']), category='initial_single_value',
            reference_group=False, summary_cell='', raw_source='Initial-submission coefficient table',
            identity_metadata='', raw_group_1based='', initial_value_retained='True'))
    evidence = json.loads((HERE.parent / 'reproducibility_overview/recipe_evidence.json').read_text(encoding='utf-8'))
    for group in evidence['groups']:
        assert group['initial']['nonzero_recipe'] == group['BO_reference']['nonzero_recipe']
    grouped = defaultdict(list)
    for row in records:
        row['source_sample'] = row['sample']
        row['sample'] = evidence['HD_aliases'].get(row['sample'], row['sample'])
        row['reference_group'] = row['sample'] in REFERENCE_GROUPS
        grouped[row['sample']].append(row)
    assert len({row['measurement_id'] for row in records}) == len(records)
    sample_rows = []
    for sample, rows in sorted(grouped.items()):
        values = [row['S_HD'] for row in rows]
        mean = stats.mean(values)
        sd = stats.stdev(values) if len(values) > 1 else None
        sample_rows.append(dict(sample=sample, n_measurements=len(values), mean_S_HD=mean,
            SD=sd, CV_percent=100 * sd / mean if sd is not None else None,
            reference_group=sample in REFERENCE_GROUPS,
            categories=';'.join(sorted({row['category'] for row in rows})),
            coefficients=';'.join(format(value, '.12g') for value in values)))
    repeated = [row for row in sample_rows if row['n_measurements'] > 1]
    refs = [row for row in repeated if row['reference_group']]
    nonrefs = [row for row in repeated if not row['reference_group']]
    assert len(records) == 806 and len(sample_rows) == 672 and len(repeated) == 103
    counts = Counter(row['n_measurements'] for row in sample_rows)
    assert counts == {1: 569, 2: 98, 8: 4, 9: 1}
    assert next(row for row in refs if row['sample'] == 'Co-Fe')['n_measurements'] == 9
    excluded = [row for row in manifest_rows if row['adopted'] != 'True']
    assert len(excluded) == 26
    categories = Counter(row['category'] for row in records)
    summary = dict(n_coefficients=806, n_sample_groups=len(sample_rows), n_repeat_groups=len(repeated),
        n_single_value_groups=counts[1], coefficients_in_repeat_groups=sum(row['n_measurements'] for row in repeated),
        replicate_counts=dict(sorted(counts.items())), source_categories=dict(categories),
        adopted_added_or_revised_coefficients=104, excluded_S_measurements=26,
        confirmed_independent_original_and_reprocessed_pairs=len(coexist),
        largest_rounding_difference=largest_rounding_difference,
        all_repeated=distribution(repeated), non_reference_pairs=distribution(nonrefs),
        reference_groups=distribution(refs),
        technical_injections=dict(n_measurements=len(injection_summary), injections_per_measurement=2,
            median_SD=stats.median(row['technical_injection_SD'] for row in injection_summary),
            median_CV_percent=stats.median(row['technical_injection_CV_percent'] for row in injection_summary)),
        grouping_note='Measurements with matching audited precursor recipes are pooled under ordinary catalyst names; each measurement is counted once.',
        recipe_evidence_sha256=sha(HERE.parent / 'reproducibility_overview/recipe_evidence.json'),
        source_policy='Adopted table values retained. No excluded measurement restored. '
                      'Technical injections are not independent experimental repeats. '
                      'Raw extraction and downstream models/figures are not modified.')
    write_csv(out / 'sample_statistics.csv', sample_rows)
    write_csv(out / 'measurement_sources.csv', records)
    write_csv(out / 'reference_groups.csv', refs)
    write_csv(out / 'largest_SD_groups.csv', sorted(repeated, key=lambda row: row['SD'], reverse=True)[:15])
    write_csv(out / 'technical_injections.csv', injection_summary)
    write_csv(out / 'replicate_counts.csv', [dict(n_measurements=n, sample_groups=count,
        coefficients=n * count, included_in_SD_distribution=n > 1) for n, count in sorted(counts.items())])
    write_csv(out / 'excluded_measurements.csv', [dict(record_id=row['record_id'], sample=row['sample'],
        S_HD=row['S_HD'], source=raw_sources[row['source']]['source'], reason=row['selection_note'])
        for row in excluded])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if args.verify_sources:
        checks = []
        for row in raw_manifest:
            if not row['snapshot'].startswith('raw/') or Path(row['snapshot']).suffix not in {'.json', '.csv'}:
                continue
            path = Path(row['source'])
            actual = sha(path) if path.exists() else ''
            checks.append(dict(source=row['source'], snapshot=row['snapshot'],
                               expected_sha256=row['sha256'], actual_sha256=actual,
                               exists=path.exists(), matches=actual == row['sha256']))
        write_csv(out / 'source_file_checks.csv', checks)
        print('Source checks:', sum(row['matches'] for row in checks), '/', len(checks))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
