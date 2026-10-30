"""SI candidates: six weighted descriptors, 15 pairs, four views per pair."""
from pathlib import Path
import hashlib
import itertools
import json
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from workflow.data import ROOT, load_data
from weighted_descriptor_helpers import weighted, joint, HELPER
from si_layout import style_joint

OUT=ROOT/'figures/si_candidates/six_weighted_descriptor_pairs'
PROPERTIES={
    'work_function':('Work function','Weighted Work Function (eV)'),
    'FE_ads_avg':('H adsorption free energy','Weighted H Adsorption Free Energy (eV)'),
    'Frequency_H':('H vibrational frequency',r'Weighted H Vibrational Frequency (cm$^{-1}$)'),
    'E_std':('Standard potential','Weighted Standard Potential (V)'),
    'pzc':('Potential of zero charge','Weighted Potential of Zero Charge (V)'),
    'delta_EOH':('OH adsorption energy','Weighted OH Adsorption Energy (eV)'),
}
Y='separation_coefficient'


def padded(values):
    lo,hi=float(np.nanmin(values)),float(np.nanmax(values))
    pad=.055*(hi-lo) if hi>lo else max(abs(lo)*.055,.01)
    return [lo-pad,hi+pad]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data,_=load_data()
    for mode in ['precursor','icp']:
        for base in PROPERTIES:
            data[mode+'_'+base]=weighted(data,base,mode)
    cols=[m+'_'+b for m in ['precursor','icp'] for b in PROPERTIES]
    data[['sample','cohort','metal_pair',Y,'nominal_frac_element1','actual_frac_element1']+cols].to_csv(OUT/'weighted_observations.csv',index=False)
    # The same descriptor has the same limits across every pair; local limits
    # enclose both representations of BO observations, without selecting on H/D.
    global_limits={};local_limits={}
    for b in PROPERTIES:
        columns=[m+'_'+b for m in ['precursor','icp']]
        global_limits['weighted_'+b]=padded(data[columns].to_numpy())
        local_limits['weighted_'+b]=padded(data.loc[data.cohort.eq('BO'),columns].to_numpy())
    joint.configure_matplotlib()
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.family':'Arial',
                         'mathtext.fontset':'custom','mathtext.rm':'Arial',
                         'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
    for b,(_,label) in PROPERTIES.items():joint.LABELS['weighted_'+b]=label
    cmap=ListedColormap(joint.BIN_COLORS);norm=BoundaryNorm(joint.BIN_EDGES,cmap.N,clip=True)
    original_contour=joint.draw_masked_knn_contour
    records=[];memberships=[];links=[];surface_files=[]
    try:
        with PdfPages(OUT/'all_15_pairs.pdf') as pdf:
            for number,(xb,yb) in enumerate(itertools.combinations(PROPERTIES,2),1):
                required=[m+'_'+b for m in ['precursor','icp'] for b in [xb,yb]]
                frame=data.dropna(subset=required+[Y]).copy()
                initial=frame.loc[frame.cohort.eq('initial')];bo=frame.loc[frame.cohort.eq('BO')]
                assert len(initial)>=36 and len(bo)>0
                stem=f'{number:02d}_{xb}__{yb}'
                memberships.extend(dict(pair=stem,sample=r.sample,cohort=r.cohort) for r in frame.itertuples())
                origin=np.array([initial[['precursor_'+b,'icp_'+b]].to_numpy().min() for b in [xb,yb]])
                span=np.array([np.ptp(initial[['precursor_'+b,'icp_'+b]].to_numpy()) for b in [xb,yb]])
                assert np.all(span>0)
                probes={}
                def contour(ax,training,x,y,xlim,ylim):
                    assert training.cohort.eq('initial').all()
                    points=training[[x,y]].to_numpy(float)
                    normalized=(points-origin)/span
                    model=KNeighborsRegressor(n_neighbors=36,weights='distance').fit(normalized,training[Y])
                    xx,yy=np.meshgrid(np.linspace(*xlim,220),np.linspace(*ylim,220))
                    query=np.c_[xx.ravel(),yy.ravel()]
                    values=model.predict((query-origin)/span)
                    axis_span=np.array([np.ptp(xlim),np.ptp(ylim)])
                    distance=NearestNeighbors(n_neighbors=1).fit(points/axis_span).kneighbors(query/axis_span)[0][:,0]
                    shown=distance<=.055
                    zz=np.ma.masked_where(~shown.reshape(xx.shape),values.reshape(xx.shape))
                    ax.contourf(xx,yy,zz,levels=joint.BIN_EDGES,cmap=cmap,norm=norm,alpha=.30,zorder=0)
                    ax.contour(xx,yy,zz,levels=joint.BIN_EDGES,colors='#777777',linewidths=.35,alpha=.30,zorder=1)
                    probe=model.predict(normalized[:5])
                    if mode in probes:np.testing.assert_allclose(probes[mode],probe,rtol=0,atol=0)
                    else:probes[mode]=probe
                    factors=np.array([1000.,.01])
                    np.testing.assert_allclose(normalized,(points*factors-origin*factors)/(span*factors),atol=1e-12)
                    name=stem+'_'+mode+'_'+view+'.npz'
                    np.savez_compressed(OUT/name,x=xx,y=yy,mean=values.reshape(xx.shape),shown=shown.reshape(xx.shape))
                    surface_files.append(name)
                    records.append(dict(pair=stem,x=xb,y=yb,representation=mode,view=view,n_initial=len(initial),n_bo=len(bo),
                        initial_visible=int(joint.filter_window(training,x,y,xlim,ylim).shape[0]),
                        supported_grid_points=int(shown.sum()),xlim=list(xlim),ylim=list(ylim),origin=origin.tolist(),span=span.tolist()))
                joint.draw_masked_knn_contour=contour
                fig=plt.figure(figsize=(10.4,8.8))
                grid=fig.add_gridspec(2,2,left=.115,right=.99,bottom=.10,top=.88,hspace=.50,wspace=.38)
                for idx,(mode,view) in enumerate([('precursor','global'),('icp','global'),('precursor','local'),('icp','local')]):
                    renamed=frame.rename(columns={mode+'_'+b:'weighted_'+b for b in [xb,yb]})
                    lims=global_limits if view=='global' else local_limits
                    x,y='weighted_'+xb,'weighted_'+yb
                    top,ax,side=joint.draw_joint_panel(fig,grid[idx//2,idx%2],renamed.loc[renamed.cohort.eq('initial')],renamed.loc[renamed.cohort.eq('BO')],x,y,lims,mode='contour')
                    style_joint(top,ax,chr(65+idx)+'  '+('Precursor' if mode=='precursor' else 'ICP')+' | '+view.title())
                title=PROPERTIES[xb][0]+' / '+PROPERTIES[yb][0]
                handles=[Patch(facecolor=c,edgecolor='black',label=l) for c,l in zip(joint.BIN_COLORS,joint.BIN_LABELS)]
                fig.legend(handles=handles,title='H/D separation coefficient',title_fontsize=14,loc='upper center',bbox_to_anchor=(.54,1.0),ncol=5,frameon=True,fancybox=False,edgecolor='black',fontsize=14)
                for ext in ['png','svg','pdf']:fig.savefig(OUT/(stem+'.'+ext),dpi=170,facecolor='white')
                pdf.savefig(fig);plt.close(fig)
                links.append((stem,title,len(initial),len(bo)))
                print(stem,len(initial),len(bo),flush=True)
    finally:joint.draw_masked_knn_contour=original_contour
    from figure_typography import normalize_exports
    from pypdf import PdfWriter
    normalize_exports(OUT,apply=True)
    combined=PdfWriter()
    for stem,_,_,_ in links:combined.append(OUT/(stem+'.pdf'))
    with (OUT/'all_15_pairs.pdf').open('wb') as output:combined.write(output)
    pd.DataFrame(records).to_csv(OUT/'panel_audit.csv',index=False)
    pd.DataFrame(memberships).to_csv(OUT/'pair_memberships.csv',index=False)
    assert len(records)==60 and len(links)==15
    files=[Path(__file__),ROOT/'weighted_descriptor_helpers.py',HELPER,ROOT/'workflow/data.py',*list((ROOT/'inputs/reference').glob('*')),*list((ROOT/'inputs/icp').glob('*.xlsx'))]
    manifest=dict(properties=list(PROPERTIES),n_pairs=15,n_panels=60,cohort_counts=data.cohort.value_counts().to_dict(),
        weighting='f1*x1+(1-f1)*x2; precursor f1=V1/(V1+V2), ICP f1=measured metal molar fraction; missing nonzero-weight properties remain missing',
        missingness='Pairwise complete across both representations; same samples for all four views of a pair; no imputation',
        background='36-neighbor inverse-distance mean of original initial H/D outcomes only; fixed pooled initial descriptor-range normalization per pair; no convex hull mask',
        display_mask='nearest initial observation within 0.055 in display-axis-normalized distance; different masks across global/local views',
        local_limits_policy='Full available BO range pooled across precursor/ICP, plus 5.5% padding; same descriptor limits across all pairs; no response-based selection',
        global_limits=global_limits,local_limits=local_limits,colors=joint.BIN_COLORS,edges=joint.BIN_EDGES.tolist(),
        tests=['60 panels and 15 pairs','at least 36 matched initial observations per pair','no BO in background','global/local probe predictions identical','distance normalization invariant to unit conversion'],
        caution='Descriptive weighted projections, not new SHAP rankings, validated surfaces, optima or mechanistic evidence. Precursor fractions retain the equal-stock-concentration assumption.',
        hashes={p.relative_to(ROOT.parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},response_or_SI_modified=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text('# Six weighted descriptor pairs for SI review\n\nRun `python plot_six_weighted_descriptor_pairs.py` from ml_revision. Fifteen pairs, each with precursor/global, ICP/global, precursor/local and ICP/local views. Arial; original discrete color palette and marginal histograms.\n\nPairwise matched observed data, not an all-six complete-case restriction. Initial H/D outcomes are original, not repeated averages. BO records are independent overlays only. See manifest.json for weighting, assumptions, fixed neighbor normalization and display-dependent masking. Local limits cover both representations of all available BO observations, plus 5.5% padding; these newly defined limits are not claimed to reproduce the three-property map limits. Values and units are inherited from the packaged elemental property table; weighted frequency is a descriptor, not a measured alloy spectrum. No ML refit or response/SI edit.\n\nEach pair has PNG/SVG/PDF and four compressed surface grids. all_15_pairs.pdf collects the figures. CSVs preserve weighted values, memberships and panel counts.\n',encoding='utf-8')
    intro='<h1>Six weighted descriptors: SI candidates</h1><p>15 pairs / 60 panels. Original H/D outcomes; precursor-volume and ICP-molar weighting. Arial; discrete colors. Pairwise complete observations without imputation. BO overlays do not enter the background.</p><p>Local views cover the pooled BO range with 5.5% padding. Blank regions are masked estimates, not low H/D separation. These plots are descriptive, not new model validation.</p><a href="all_15_pairs.pdf">All 15 figures (PDF)</a>'
    nav='<ul>'+''.join(f'<li><a href="#{s}">{html.escape(t)}</a> ({n} initial + {b} BO)</li>' for s,t,n,b in links)+'</ul>'
    panels=''.join(f'<section id="{s}"><h2>{html.escape(t)}</h2><a href="{s}.svg">SVG</a> | <a href="{s}.pdf">PDF</a><img loading="lazy" src="{s}.png"></section>' for s,t,_,_ in links)
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Weighted descriptor pairs</title><style>body{font-family:Arial;margin:24px;max-width:1400px}img{width:100%}section{margin:40px 0}li{margin:6px 0}</style>'+intro+nav+panels,encoding='utf-8')


if __name__=='__main__':main()
