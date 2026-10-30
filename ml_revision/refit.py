"""Standalone nested refits with training-only correlation-aware selection."""
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import r2_score, mean_squared_error
from cohorts import KEYS, prepare
from descriptors import DescriptorTransform
from model_core import digest, estimator, save
from selector import fit_selector, choose_candidates, correlation_groups, permutation_utilities
from workflow.data import ROOT


def cached_selector(path, frame, job, config, selection, seed, fingerprint):
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['fingerprint'] == fingerprint
        assert record['fitting_samples'] == frame['sample'].tolist()
        return record
    record = fit_selector(frame, job['protocol'], selection['threshold'], config, selection, seed, job['scheme'])
    record.update(fingerprint=fingerprint, seed=seed, scheme=job['scheme'])
    save(path, record)
    return record


def final_fit(job, choice, mask, strategy, config, selection):
    t = DescriptorTransform(job['scheme']).fit(job['train'])
    x, v = (t.transform(job[k]).to_numpy() for k in ('train', 'test'))
    m = estimator(choice['model'], choice['config'], config['final_trees'], job['seed']).fit(x[:, mask], job['train'].target)
    pred = m.predict(v[:, mask])
    explanation = shap.TreeExplainer(m)
    phi = np.asarray(explanation.shap_values(v[:, mask], check_additivity=False))
    baseline = float(np.asarray(explanation.expected_value).ravel()[0])
    np.testing.assert_allclose(phi.sum(axis=1)+baseline, pred, atol=1e-6, rtol=1e-6)
    full = np.zeros(v.shape)
    full[:, mask] = phi
    groups, corr = correlation_groups(x, selection['threshold'])
    active = [(g, [mask.index(i) for i in g if i in mask]) for g in groups if set(g) & set(mask)]
    importance, sd = permutation_utilities(m, v[:, mask], job['test'].target, [local for _, local in active],
                                           selection['heldout_permutations'], job['seed']+12001)
    grouped = [dict(features=[t.features[i] for i in g], selected_features=[t.features[mask[i]] for i in local],
                    permutation_delta_r2=float(importance[k]), permutation_sd=float(sd[k]),
                    mean_absolute_sum_shap=float(np.abs(full[:, g].sum(axis=1)).mean()))
               for k, (g, local) in enumerate(active)]
    pearson = pd.DataFrame(x).corr().fillna(0).to_numpy()
    pairs = [dict(feature_a=t.features[i], feature_b=t.features[k], pearson=float(pearson[i, k]),
                  spearman=float(corr[i, k])) for a, i in enumerate(mask) for k in mask[a+1:]
             if max(abs(pearson[i, k]), abs(corr[i, k])) >= .8]
    return dict(strategy=strategy, features=[t.features[i] for i in mask], columns=t.features, mask=mask,
                model=choice['model'], config=choice['config'], seed=job['seed'],
                requested_k=choice['k'], actual_k=len(mask), inner_mean_r2=choice['mean'],
                inner_sd_r2=choice['sd'], medians=t.medians, base_value=baseline,
                predictions=[dict(sample=s, y_true=float(y), y_pred=float(p)) for s, y, p in
                             zip(job['test']['sample'], job['test'].target, pred)],
                shap=full.tolist(), feature_values=v.tolist(), grouped_importance=grouped, correlated_pairs=pairs)


def train(job, config, selection, output, fingerprint):
    key = {k: job[k] for k in KEYS}
    name = '_'.join(str(v) for v in key.values())
    checkpoint = output / 'checkpoints' / (name+'.json')
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text(encoding='utf-8'))
        assert result['fingerprint'] == fingerprint
        return result
    start = time.perf_counter()
    directory = output / 'selection' / name
    inner = []
    for i, (a, b) in enumerate(job['inner']):
        selected = cached_selector(directory / f'inner_{i}.json', a, job, config, selection,
                                   job['seed']+i*10000, fingerprint)
        assert not set(selected['fitting_samples']) & (set(b['sample']) | set(job['test']['sample']))
        t = DescriptorTransform(job['scheme']).fit(a)
        inner.append((t.transform(a).to_numpy(), t.transform(b).to_numpy(), a.target.to_numpy(),
                      b.target.to_numpy(), selected['ranked']))
    maximum = min(len(item[4]) for item in inner)
    counts = sorted(set([min(k, maximum) for k in selection['candidate_counts']] + [maximum]))
    tune_path = directory / 'tuning.json'
    if tune_path.exists():
        tuning = json.loads(tune_path.read_text(encoding='utf-8'))
        assert tuning['fingerprint'] == fingerprint
    else:
        candidates = []
        for strategy in ['correlation90', 'all_candidates']:
            for k in (counts if strategy == 'correlation90' else [inner[0][0].shape[1]]):
                for model in config['models']:
                    for ci, parameters in enumerate(config['configs']):
                        scores = []
                        for i, (x, v, y, z, ranking) in enumerate(inner):
                            mask = ranking[:k] if strategy == 'correlation90' else list(range(k))
                            fit = estimator(model, parameters, config['tuning_trees'], job['seed']+i).fit(x[:, mask], y)
                            scores.append(float(r2_score(z, fit.predict(v[:, mask]))))
                        candidates.append(dict(strategy=strategy, k=k, model=model, config_id=ci, config=parameters,
                                               scores=scores, mean=float(np.mean(scores)), sd=float(np.std(scores, ddof=1))))
        tuning = dict(fingerprint=fingerprint, candidates=candidates)
        save(tune_path, tuning)
    selected = cached_selector(directory / 'outer_train.json', job['train'], job, config, selection,
                               job['seed']+90000, fingerprint)
    assert not set(selected['fitting_samples']) & set(job['test']['sample'])
    fitted = []
    for strategy in ['correlation90', 'all_candidates']:
        choices = [r for r in tuning['candidates'] if r['strategy'] == strategy]
        best = choose_candidates(choices, config['inner_folds'])[0]
        mask = selected['ranked'][:best['k']] if strategy == 'correlation90' else list(range(best['k']))
        fitted.append(final_fit(job, best, mask, strategy, config, selection))
    result = dict(**key, scheme=job['scheme'], fingerprint=fingerprint, fitting=fitted,
                  elapsed_seconds=time.perf_counter()-start)
    save(checkpoint, result)
    print(name, 'COMPLETE', round(result['elapsed_seconds']), 's', 'k='+str(fitted[0]['actual_k']), flush=True)
    return result


def summarize(results, output):
    predictions, settings, rows, groups, pairs = [], [], [], [], []
    for r in results:
        key = {k: r[k] for k in KEYS}
        for f in r['fitting']:
            context = dict(**key, strategy=f['strategy'])
            predictions += [dict(**context, **p) for p in f['predictions']]
            settings.append(dict(**context, features=f['features'], n_features=f['actual_k'],
                                 requested_k=f['requested_k'], model=f['model'], config=f['config'], seed=f['seed']))
            for i, p in enumerate(f['predictions']):
                for j, feature in enumerate(f['columns']):
                    rows.append(dict(**context, sample=p['sample'], feature=feature, value=f['shap'][i][j],
                                     feature_value=f['feature_values'][i][j], selected=feature in f['features']))
            groups += [dict(**context, **g) for g in f['grouped_importance']]
            pairs += [dict(**context, **p) for p in f['correlated_pairs']]
    for name, records in [('predictions', predictions), ('shap', rows), ('grouped_importance', groups), ('correlated_pairs', pairs)]:
        pd.DataFrame(records).to_csv(output / (name+'.csv'), index=False)
    save(output / 'settings.json', settings)
    metrics = []
    for key, g in pd.DataFrame(predictions).groupby(['workflow', 'protocol', 'arm', 'strategy', 'repeat']):
        assert g['sample'].is_unique
        metrics.append(dict(zip(['workflow', 'protocol', 'arm', 'strategy', 'repeat'], key), n=len(g),
                            r2=float(r2_score(g.y_true, g.y_pred)), rmse=float(np.sqrt(mean_squared_error(g.y_true, g.y_pred)))))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / 'repeat_metrics.csv', index=False)
    metrics.groupby(['workflow', 'protocol', 'arm', 'strategy'], as_index=False).agg(n=('n', 'min'),
        repeats=('repeat', 'size'), r2_mean=('r2', 'mean'), r2_sd=('r2', 'std'), rmse_mean=('rmse', 'mean'),
        rmse_sd=('rmse', 'std')).to_csv(output / 'summary.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workflow', choices=['hd_icp', 'hd_precursor', 'hd_fraction', 'ec_unshifted_measured', 'ec_shifted', 'ec_shifted_measured', 'eta10'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    selection = json.loads((ROOT / 'selection_config.json').read_text(encoding='utf-8'))
    output = ROOT / ('smoke' if args.smoke else 'results')
    jobs = prepare(config, output)
    expected = len(jobs)
    if args.prepare_only:
        print('Verified 285 jobs, exact locked outer and inner memberships, and both eta10 targets.', flush=True)
        return
    if args.workflow:
        jobs = [j for j in jobs if j['workflow'] == args.workflow]
    if args.smoke:
        seen = set()
        subset = []
        for j in jobs:
            key = (j['workflow'], j['protocol'], j['arm'])
            if key not in seen:
                subset.append(j)
                seen.add(key)
        jobs = subset
        selection.update(resamples=4, ranking_permutations=2, candidate_counts=[3, 10], heldout_permutations=2)
        config.update(selection_trees=8, tuning_trees=5, final_trees=10)
        config['configs'] = config['configs'][:1]
    if args.limit:
        jobs = jobs[:args.limit]
    files = [*sorted((ROOT / 'inputs').rglob('*')), *sorted((ROOT / 'workflow').glob('*.py')),
             *[ROOT / n for n in ['refit.py', 'selector.py', 'descriptors.py', 'cohorts.py', 'model_core.py', 'config.json', 'selection_config.json', 'audit_electrochemical.py']]]
    manifest = dict(config=config, selection=selection, smoke=args.smoke,
        hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files if p.is_file()},
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__, shap=shap.__version__),
        methods='Fold-local complete-linkage Spearman groups, resampled representative ranking, inner-best subset size.',
        comparison_inputs_used_in_fitting=False)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if (output / 'manifest.json').exists():
        assert json.loads((output / 'manifest.json').read_text(encoding='utf-8')) == manifest, 'Fit inputs or code changed'
    save(output / 'manifest.json', manifest)
    save(output / 'verification.json', dict(complete=False, scientific=False, expected_jobs=expected, scheduled_jobs=len(jobs)))
    start = time.perf_counter()
    with parallel_config(backend='loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs)(delayed(train)(j, config, selection, output, fingerprint) for j in jobs)
    summarize(results, output)
    for name, value in manifest['hashes'].items():
        assert digest(ROOT / name) == value, name
    complete = len(jobs) == expected and not args.smoke
    save(output / 'verification.json', dict(complete=complete, scientific=False,
         expected_jobs=expected, completed_jobs=len(results), elapsed_seconds=time.perf_counter()-start,
         pending='Independent validation and deterministic refit checks'))
    print('Finished', len(results), 'jobs.', flush=True)


if __name__ == '__main__':
    main()
