"""Audit all folds, selectors, predictions and deterministic final refits."""
import argparse
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from cohorts import KEYS, prepare
from descriptors import DescriptorTransform
from model_core import digest, estimator, save
from selector import correlation_groups, choose_candidates, fit_selector
from workflow.data import ROOT
from workflow.recipes import RecipeGuard


def validate_job(job, result, output, config, selection, fingerprint, refit_models):
    name = '_'.join(str(job[k]) for k in KEYS)
    assert result['fingerprint'] == fingerprint
    directory = output / 'selection' / name
    allowed = set(job['train']['sample'])
    forbidden = set(job['test']['sample'])
    for filename, frame in [(f'inner_{i}.json', a) for i,(a,b) in enumerate(job['inner'])] + [('outer_train.json', job['train'])]:
        record = json.loads((directory / filename).read_text())
        assert record['fingerprint'] == fingerprint
        assert record['fitting_samples'] == frame['sample'].tolist()
        assert set(record['fitting_samples']) <= allowed
        assert not set(record['fitting_samples']) & forbidden
        transform = DescriptorTransform(job['scheme']).fit(frame)
        assert transform.medians == record['medians']
        groups, corr = correlation_groups(transform.transform(frame).to_numpy(), selection['threshold'])
        assert record['groups'] == groups
        np.testing.assert_allclose(record['correlation'],corr,atol=1e-12,rtol=0)
        assert len(record['ranked'])==len(groups)
        for g in groups:
            assert len(set(record['ranked'])&set(g))==1
        guard = RecipeGuard(frame)
        for item in record['resample_memberships']:
            a,b = item['train'],item['validation']
            assert set(a)|set(b)==set(frame['sample'])
            assert not set(a)&set(b)
            guard.check(frame.loc[a],frame.loc[b],'validation of selector resample')
            if job['protocol']=='unseen_pair':
                assert not set(frame.loc[a].metal_pair)&set(frame.loc[b].metal_pair)
    outer = json.loads((directory/'outer_train.json').read_text())
    tuning = json.loads((directory/'tuning.json').read_text())
    maximum = 0.
    for f in result['fitting']:
        candidates = [c for c in tuning['candidates'] if c['strategy']==f['strategy']]
        best = choose_candidates(candidates,config['inner_folds'])[0]
        expected_mask = outer['ranked'][:best['k']] if f['strategy']=='correlation90' else list(range(best['k']))
        assert f['mask']==expected_mask and f['model']==best['model'] and f['config']==best['config']
        transform = DescriptorTransform(job['scheme']).fit(job['train'])
        x,v = [transform.transform(job[k]).to_numpy() for k in ['train','test']]
        assert f['columns']==transform.features
        np.testing.assert_allclose(v,f['feature_values'],atol=0,rtol=0)
        predictions = np.array([p['y_pred'] for p in f['predictions']])
        np.testing.assert_array_equal([p['sample'] for p in f['predictions']],job['test']['sample'])
        np.testing.assert_allclose([p['y_true'] for p in f['predictions']],job['test'].target,atol=0,rtol=0)
        phi = np.array(f['shap'])
        np.testing.assert_allclose(phi.sum(axis=1)+f['base_value'],predictions,atol=1e-6,rtol=1e-6)
        unselected = [i for i in range(v.shape[1]) if i not in f['mask']]
        assert not np.any(phi[:,unselected])
        if refit_models:
            model = estimator(f['model'],f['config'],config['final_trees'],job['seed']).fit(x[:,f['mask']],job['train'].target)
            fresh = model.predict(v[:,f['mask']])
            np.testing.assert_allclose(fresh,predictions,atol=1e-10,rtol=0)
            maximum = max(maximum,float(np.max(np.abs(fresh-predictions))))
    return dict(**{k:job[k] for k in KEYS},selectors=4,models=2,final_refits=2 if refit_models else 0,
                max_prediction_difference=maximum)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--jobs',type=int,default=6)
    parser.add_argument('--check-only',action='store_true',help='Validate saved records without fitting any estimator or selector.')
    args=parser.parse_args()
    output=ROOT/'results'
    manifest=json.loads((output/'manifest.json').read_text())
    fingerprint=__import__('hashlib').sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    for name,sha in manifest['hashes'].items():
        assert digest(ROOT/name)==sha,name
    config,selection=manifest['config'],manifest['selection']
    jobs=prepare(config)
    records=[]
    for j in jobs:
        name='_'.join(str(j[k]) for k in KEYS)
        records.append(json.loads((output/'checkpoints'/(name+'.json')).read_text()))
    with parallel_config(backend='loky',inner_max_num_threads=1):
        checks=Parallel(n_jobs=args.jobs)(delayed(validate_job)(j,r,output,config,selection,fingerprint,not args.check_only) for j,r in zip(jobs,records))
    if not args.check_only:
        save(output/'fold_validation.json',checks)
    pd.testing.assert_frame_equal(pd.read_csv(ROOT/'inputs/paired/sample_covariates.csv'),
                                  pd.read_csv(output/'electrochemical_audit/sample_covariates.csv'),check_exact=True)
    deterministic=[]
    for scheme in ([] if args.check_only else ['icp','precursor','fraction','unshifted_measured','shifted','shifted_measured']):
        j=next(j for j in jobs if j['scheme']==scheme)
        name='_'.join(str(j[k]) for k in KEYS)
        recorded=json.loads((output/'selection'/name/'inner_0.json').read_text())
        fresh=fit_selector(j['inner'][0][0],j['protocol'],selection['threshold'],config,selection,j['seed'],j['scheme'])
        assert fresh=={k:v for k,v in recorded.items() if k not in ['fingerprint','seed','scheme']}
        deterministic.append(scheme)
    verification=dict(complete=True,scientific=True,completed_jobs=len(records),
        models_checked=sum(c['models'] for c in checks),selectors_checked=sum(c['selectors'] for c in checks),
        deterministic_final_refits=sum(c['final_refits'] for c in checks),
        max_prediction_difference=max(c['max_prediction_difference'] for c in checks),
        deterministic_full_selector_checks=deterministic,check_only=args.check_only,
        all_outer_inner_folds_match=True,raw_paired_covariates_match=True,hashes_verified=len(manifest['hashes']))
    destination = ROOT/'checks/saved_records.json' if args.check_only else output/'verification.json'
    save(destination,verification)
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':main()
