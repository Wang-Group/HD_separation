"""Pairwise-complete observed correlations; no fitted model or imputation."""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from plot_helpers import ROOT, observed, compact_label
from descriptors import COVARIATES
from workflow.data import FEATURES, load_data
from model_core import save

def main():
    out=ROOT/'descriptive/observed_correlations';out.mkdir(parents=True,exist_ok=True)
    data,_=load_data(); data=data.loc[data.cohort.eq('initial')]
    frame=observed('icp').reindex(data.index)
    targets=pd.read_csv(ROOT/'results/eta10_target_audit.csv').set_index('sample')
    frame['eta10']=targets.original_eta10_mV.where(targets.original_status.eq('eta10_crossing'))
    frame['separation_coefficient']=data.separation_coefficient
    cov=pd.read_csv(ROOT/'inputs/paired/sample_covariates.csv').set_index('sample')
    frame[COVARIATES]=cov[COVARIATES].reindex(frame.index)
    assert frame.eta10.notna().sum()==298 and frame.eta10.gt(550).sum()==18
    outcomes=['separation_coefficient','eta10']+COVARIATES
    pairs=[(a,b) for a in FEATURES for b in outcomes]+[(a,'separation_coefficient') for a in ['eta10']+COVARIATES]
    rows=[]
    for a,b in pairs:
        g=frame[[a,b]].dropna()
        for name,fn in [('pearson',pearsonr),('spearman',spearmanr)]:
            r,p=fn(g[a],g[b]);rows.append(dict(feature=a,outcome=b,method=name,n=len(g),r=r,p_nominal=p))
    corr=pd.DataFrame(rows)
    frame.to_csv(out/'observations.csv');corr.to_csv(out/'correlations.csv',index=False)
    table=corr.loc[corr.feature.isin(['eta10']+COVARIATES)]
    table.to_csv(ROOT/'tables/electrochemical_outcome_correlations.csv',index=False)
    plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,
        'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
    fig,axes=plt.subplots(1,2,figsize=(9,9),sharey=True)
    fig.subplots_adjust(left=.30,right=.88,top=.95,bottom=.18,wspace=.12)
    for ax,method in zip(axes,['pearson','spearman']):
        m=corr.loc[corr.feature.isin(FEATURES)&corr.method.eq(method)].pivot(index='feature',columns='outcome',values='r').reindex(index=FEATURES,columns=outcomes)
        image=ax.imshow(m,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
        for i,j in np.ndindex(m.shape):ax.text(j,i,f'{m.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9.5,color='white' if abs(m.iloc[i,j])>.65 else '#222')
        ax.set_yticks(range(30),[compact_label(f) for f in FEATURES],fontsize=11)
        ax.set_xticks(range(5),['H/D coefficient',r'Initial LSV $\eta_{10}$','Current near -1 V','EIS resistance','Corrected potential'],rotation=60,ha='right',fontsize=11)
        ax.set_title(method.title(),fontweight='bold',fontsize=13)
    cax=fig.add_axes([.90,.25,.020,.6]);fig.colorbar(image,cax=cax,label='Correlation',ticks=[-1,-.5,0,.5,1])
    cax.tick_params(labelsize=11,pad=2);cax.yaxis.label.set_size(12);cax.yaxis.labelpad=3
    for ext in ['png','svg','pdf']:fig.savefig(out/f'descriptor_outcome_correlations.{ext}',dpi=200)
    plt.close(fig)
    save(out/'verification.json',dict(initial=560,eta10=298,above_550_retained=18,descriptors=30,
        imputation=False,target_cap=False,paired_measurements='inputs/paired/sample_covariates.csv'))
    print(out)

if __name__=='__main__':main()
