"""Check publication dependencies and recompute reporting metrics without fitting."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from model_core import digest, save
from workflow.data import ROOT, load_data
from cohorts import eta_targets
from figure_typography import normalize_exports

def main():
    manifest=json.loads((ROOT/'results/manifest.json').read_text())
    for name,sha in manifest['hashes'].items():assert digest(ROOT/name)==sha,name
    pred=pd.read_csv(ROOT/'results/predictions.csv')
    metrics=pd.read_csv(ROOT/'results/repeat_metrics.csv')
    keys=['workflow','protocol','arm','strategy','repeat']
    rebuilt=[]
    for key,g in pred.groupby(keys):
        assert g['sample'].is_unique
        rebuilt.append(dict(zip(keys,key),r2=r2_score(g.y_true,g.y_pred),rmse=np.sqrt(mean_squared_error(g.y_true,g.y_pred))))
    fresh=pd.DataFrame(rebuilt).set_index(keys).sort_index()
    expected=metrics.set_index(keys).sort_index()
    np.testing.assert_allclose(fresh[['r2','rmse']],expected[['r2','rmse']],rtol=0,atol=1e-10)
    summary=pd.read_csv(ROOT/'results/summary.csv').set_index(keys[:-1]).sort_index()
    for key,g in fresh.reset_index().groupby(keys[:-1]):
        row=summary.loc[key]
        for measure in ['r2','rmse']:
            np.testing.assert_allclose([g[measure].mean(),g[measure].std()],row[[measure+'_mean',measure+'_sd']],atol=1e-10,rtol=0)
    data,_=load_data();targets=eta_targets(data)
    pd.testing.assert_frame_equal(targets,pd.read_csv(ROOT/'results/eta10_target_audit.csv'),check_dtype=False,check_exact=False,atol=1e-10,rtol=0)
    cov=pd.read_csv(ROOT/'inputs/paired/sample_covariates.csv')
    pd.testing.assert_frame_equal(cov,pd.read_csv(ROOT/'results/electrochemical_audit/sample_covariates.csv'),check_exact=True)
    assert not normalize_exports(ROOT/'figures')
    oversized=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.stat().st_size>100*1024**2]
    assert not oversized,oversized
    index=json.loads((ROOT/'FIGURE_INDEX.json').read_text())
    for name,sha in index['hashes'].items():assert digest(ROOT/name)==sha,name
    compressed=ROOT/'results/shap.csv.gz'
    if compressed.exists():
        with gzip.open(compressed,'rb') as f:payload_hash=hashlib.file_digest(f,'sha256').hexdigest()
        record=json.loads((ROOT/'checks/compression.json').read_text())
        assert payload_hash==record['uncompressed_sha256']
    report=dict(passed=True,fit_performed=False,input_and_code_hashes=len(manifest['hashes']),
        repeat_metrics=len(fresh),summary_rows=len(summary),initial_samples=560,eligible_bo=111,
        eta10_samples=int(targets.included.sum()),eta10_crossings=int(targets.loc[targets.included,'n_accepted'].sum()),
        raw_paired_inputs_unchanged=True,oversized_files=oversized,figure_hashes=len(index['hashes']))
    save(ROOT/'checks/publication.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
