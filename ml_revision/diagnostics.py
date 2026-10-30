"""Descriptive selection stability and residual-correlation diagnostics."""
import json
import numpy as np
import pandas as pd
from workflow.data import ROOT
from descriptors import DescriptorTransform


def stability(masks,features):
    matrix=np.array([[f in mask for f in features] for mask in masks],dtype=float)
    q=float(matrix.mean())
    if q<=0 or q>=1:return np.nan
    return float(1-matrix.var(axis=0,ddof=1).mean()/(q*(1-q)))


def run():
    results=ROOT/'results'
    settings=pd.DataFrame(json.loads((results/'settings.json').read_text()))
    pairs=pd.read_csv(results/'correlated_pairs.csv')
    rows=[]
    schemes={'hd_icp':'icp','hd_precursor':'precursor','hd_fraction':'fraction','ec_unshifted_measured':'unshifted_measured',
             'ec_shifted':'shifted','ec_shifted_measured':'shifted_measured','eta10':'icp'}
    for (wf,protocol,arm,strategy),g in settings.groupby(['workflow','protocol','arm','strategy']):
        assert len(g)==15
        f=pairs.loc[pairs.workflow.eq(wf)&pairs.protocol.eq(protocol)&pairs.arm.eq(arm)&pairs.strategy.eq(strategy)]
        counts08=[];counts09=[];fractions=[]
        for r in g.itertuples():
            local=f.loc[f.repeat.eq(r.repeat)&f.fold.eq(r.fold)]
            counts08.append(len(local))
            counts09.append(int((local.pearson.abs().ge(.9)|local.spearman.abs().ge(.9)).sum()))
            total=r.n_features*(r.n_features-1)/2
            fractions.append(len(local)/total if total else 0.)
        rows.append(dict(workflow=wf,protocol=protocol,arm=arm,strategy=strategy,
            mean_k=g.n_features.mean(),min_k=int(g.n_features.min()),max_k=int(g.n_features.max()),
            stability=stability(g.features,DescriptorTransform(schemes[wf]).features),
            median_pairs_08=float(np.median(counts08)),median_pairs_09=float(np.median(counts09)),
            mean_pair_fraction_08=float(np.mean(fractions))))
    out=pd.DataFrame(rows)
    (ROOT/'tables').mkdir(exist_ok=True)
    out.to_csv(ROOT/'tables/selection_diagnostics.csv',index=False)
    return out


if __name__=='__main__':run()
