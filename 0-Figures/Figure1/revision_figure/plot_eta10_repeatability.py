"""Descriptive repeatability of uncapped accepted eta10 crossings; no fitting."""
from pathlib import Path
import hashlib
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lsv_plot_data import load_data, input_hashes

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'repeatability_review'


def main():
    OUT.mkdir(exist_ok=True)
    curves,samples=load_data()
    samples.to_csv(OUT/'plot_sample_statistics.csv',index=False)
    accepted=curves.loc[curves.status.eq('eta10_crossing')].copy()
    rebuilt=accepted.groupby('sample').eta10_mV.agg(['mean','std','size'])
    given=samples.set_index('sample').loc[rebuilt.index]
    np.testing.assert_allclose(rebuilt['mean'],given.eta10_mean_mV,atol=1e-9)
    np.testing.assert_allclose(rebuilt['std'],given.eta10_SD_mV,atol=1e-9,equal_nan=True)
    np.testing.assert_array_equal(rebuilt['size'],given.n_eta10)
    selected=samples.loc[samples.n_eta10.ge(2)].copy()
    selected['group']=np.select([selected['sample'].str.startswith('reference_'),selected['sample'].str.contains(r'\d+$')],['Reference groups','Numbered BO samples'],default='Unnumbered samples')
    selected=selected.sort_values('eta10_SD_mV',ascending=False)
    selected.to_csv(OUT/'sample_SD_distribution.csv',index=False)
    accepted.loc[accepted['sample'].isin(selected['sample'])].to_csv(OUT/'accepted_measurements.csv',index=False)
    selected.head(20).to_csv(OUT/'largest_20_SDs.csv',index=False)
    sd=selected.eta10_SD_mV
    summary=dict(n_sample_groups=len(sd),n_accepted_measurements=int(selected.n_eta10.sum()),
        n_single_crossing_groups=int(samples.n_eta10.eq(1).sum()),n_no_crossing_groups=int(samples.n_eta10.eq(0).sum()),
        mean_SD_mV=float(sd.mean()),median_SD_mV=float(sd.median()),
        Q1_mV=float(sd.quantile(.25)),Q3_mV=float(sd.quantile(.75)),
        P90_mV=float(sd.quantile(.90)),P95_mV=float(sd.quantile(.95)),max_SD_mV=float(sd.max()))
    thresholds=pd.DataFrame([dict(threshold_mV=t,n_at_or_below=int(sd.le(t).sum()),percent_at_or_below=100*sd.le(t).mean(),n_above=int(sd.gt(t).sum())) for t in [10,20,30,50,75,100]])
    thresholds.to_csv(OUT/'thresholds.csv',index=False)
    summary.update(total_curves=len(curves), total_samples=len(samples),
        status_counts=curves.status.value_counts().to_dict(),
        n_accepted_eta10=int(curves.status.eq('eta10_crossing').sum()),
        sample_metric_counts=samples.potential_metric.value_counts().to_dict(),
        normalization_counts=curves.normalization_basis.value_counts().to_dict(),
        thresholds=thresholds.to_dict('records'), grouping='Audited same-recipe measurements pooled once')
    counts_all=samples.n_eta10.value_counts().sort_index()
    pd.DataFrame([dict(n_accepted_crossings=n,sample_groups=count,
        accepted_measurements=n*count,included_in_SD_distribution=n>=2)
        for n,count in counts_all.items()]).to_csv(OUT/'replicate_counts.csv',index=False)
    groups=selected.groupby('group',as_index=False).agg(n=('sample','size'),mean_SD_mV=('eta10_SD_mV','mean'),median_SD_mV=('eta10_SD_mV','median'),max_SD_mV=('eta10_SD_mV','max'))
    groups.to_csv(OUT/'group_summary.csv',index=False)
    selected.groupby('n_eta10').agg(n_sample_groups=('sample','size'),median_SD_mV=('eta10_SD_mV','median')).to_csv(OUT/'replicate_count_summary.csv')
    plt.rcParams.update({'font.family':'Arial','font.size':12,'svg.fonttype':'none',
        'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic',
        'mathtext.bf':'Arial:bold','pdf.fonttype':42,'axes.labelsize':13})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    fig.subplots_adjust(left=.075,right=.98,bottom=.17,top=.83,wspace=.25)
    bins=np.arange(0,np.ceil(sd.max()/10)*10+10,10)
    counts,_=np.histogram(sd,bins=bins)
    pd.DataFrame({'bin_start_mV':bins[:-1],'bin_end_mV':bins[1:],'count':counts}).to_csv(OUT/'histogram_bins.csv',index=False)
    axes[0].hist(sd,bins=bins,color='#26849c',edgecolor='white',linewidth=.6)
    axes[0].axvline(sd.median(),color='#b44a3a',linestyle='--',lw=1.5,label=f'Median: {sd.median():.2f} mV')
    axes[0].axvline(sd.mean(),color='#343434',linestyle=':',lw=1.5,label=f'Mean: {sd.mean():.2f} mV')
    axes[0].set(ylabel='Number of catalysts',xlabel=r'Sample SD of $\eta_{10}$ (mV)',xlim=(0,bins[-1]))
    axes[0].legend(frameon=False,fontsize=11)
    x=np.sort(sd.to_numpy());y=np.arange(1,len(x)+1)/len(x)*100
    axes[1].step(np.r_[0,x],np.r_[0,y],where='post',color='#26849c',lw=1.8)
    for threshold,color,dy in [(30,'#b44a3a',-11),(50,'#555555',-12)]:
        percent=100*sd.le(threshold).mean()
        axes[1].plot([threshold,threshold],[0,percent],ls='--',color=color,lw=1)
        axes[1].plot(threshold,percent,'o',color=color,ms=4)
        axes[1].annotate(f'{percent:.1f}% at or below {threshold} mV',(threshold,percent),xytext=(threshold+9,percent+dy),fontsize=11,color=color)
    axes[1].set(ylabel='Cumulative proportion (%)',xlabel=r'Sample SD of $\eta_{10}$ (mV)',xlim=(0,bins[-1]),ylim=(0,103))
    for i,ax in enumerate(axes):
        ax.spines[['top','right']].set_visible(False)
        ax.text(-.12,.99,'AB'[i],transform=ax.transAxes,fontweight='bold',fontsize=17,va='top')
    fig.suptitle(r'Experimental variability of $\eta_{10}$',fontsize=16,y=.98)
    fig.text(.5,.88,f'{len(selected)} catalysts with at least two accepted crossings; uncapped values',ha='center',fontsize=12)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/('eta10_SD_distribution.'+ext),dpi=200)
    plt.close(fig)
    caption=(f'Distribution of sample standard deviations of accepted crossing-based eta10 values for {len(selected)} sample groups '
        f'with at least two measurements. SD was calculated with the n-1 denominator from uncapped per-curve values; '
        f'each sample group contributes one SD. Median SD: {sd.median():.2f} mV; interquartile range: {sd.quantile(.25):.2f}-{sd.quantile(.75):.2f} mV; mean SD: {sd.mean():.2f} mV. '
        'All retained high-SD groups are included. The left panel uses 10-mV histogram bins; the right panel shows the empirical cumulative distribution.')
    notes=[caption,
        'Scope: all catalysts in the current Figure 1 extraction, not only the ICP-eligible ML cohort. Measurements with matching audited precursor recipes are pooled under ordinary catalyst names, each counted once.',
        f'{summary["n_single_crossing_groups"]} single-crossing groups have undefined SD and are excluded, not assigned zero. {summary["n_no_crossing_groups"]} groups have no accepted eta10 crossing.',
        'Rising-point and nearest-point estimates, scan-limit bounds and no-result placeholders are excluded from numerical eta10 SDs. Samples with mixed statuses contribute their accepted crossings only; excluded-status counts are retained in the sample table.',
        'The 10-550 mV ML cap and graphical boundary clipping are not applied to experimental means or SDs. Geometric area normalization follows the verified extraction, including 2 cm2 curves.',
        'These are across-retained-measurement SDs. Acquisition conditions can differ (including scan rate), so this is not a claim of identical-condition analytical precision. SD estimates with n=2 are less stable than those from larger replicate counts. No curves were removed to narrow this distribution.',
        'Extraction uses the common automatic rising-point rule with one explicitly recorded Co-Ce repeat4 noise exception. This script does not fit models or change Figure 4.']
    (OUT/'README.md').write_text('# Eta10 repeatability: descriptive review\n\n'+'\n\n'.join(notes)+'\n\nRun `python plot_eta10_repeatability.py` from revision_figure.\n',encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    hashes={**input_hashes(),Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(dict(means_and_sample_SDs_rebuilt=True,hashes=hashes),indent=2)+'\n',encoding='utf-8')
    page='<html lang="en"><meta charset="utf-8"><title>Eta10 repeatability</title><style>body{font:17px Georgia,serif;max-width:1200px;margin:30px auto;padding:0 20px}img{width:100%}td,th{padding:7px 12px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>Eta10 repeatability</h1><img src="eta10_SD_distribution.png" alt="SD histogram and cumulative distribution">'
    page+='<p>'+html.escape(caption)+'</p><h2>Thresholds</h2>'+thresholds.to_html(index=False,float_format=lambda x:f'{x:.2f}')
    page+='<h2>Groups</h2>'+groups.to_html(index=False,float_format=lambda x:f'{x:.2f}')
    page+='<h2>Largest SDs</h2>'+selected[['sample','n_eta10','eta10_mean_mV','eta10_SD_mV']].head(20).to_html(index=False,float_format=lambda x:f'{x:.2f}')
    page+='<h2>Scope and interpretation</h2><ul>'+''.join('<li>'+html.escape(n)+'</li>' for n in notes[1:])+'</ul></html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,indent=2));print(thresholds.to_string(index=False));print(groups.to_string(index=False))


if __name__=='__main__':main()
