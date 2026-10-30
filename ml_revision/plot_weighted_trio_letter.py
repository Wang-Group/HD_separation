"""Reproduce precursor- and ICP-weighted three-property maps."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from workflow.data import ROOT, load_data
from weighted_descriptor_helpers import weighted, joint, HELPER
from plot_six_weighted_descriptor_pairs import PROPERTIES, padded
from si_layout import style_joint

BASES=['work_function','FE_ads_avg','electronegativity']
PROPERTIES = dict(PROPERTIES, electronegativity=('Electronegativity','Weighted Electronegativity'))
OUT=ROOT/'figures/weighted_descriptor_maps'
Y='separation_coefficient'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data,_=load_data();columns=[]
    for mode in ['precursor','icp']:
        for b in BASES:
            c=mode+'_'+b;data[c]=weighted(data,b,mode);columns.append(c)
    # Match the complete cohort used to compare these maps with the frequency maps.
    required = columns + [Y]
    for mode in ['precursor','icp']:
        c=mode+'_Frequency_H'
        data[c]=weighted(data,'Frequency_H',mode)
        required.append(c)
    data=data.dropna(subset=required).copy()
    initial=data.loc[data.cohort.eq('initial')];bo=data.loc[data.cohort.eq('BO')]
    assert len(initial)>=36 and len(bo)==111
    data[['sample','cohort','metal_pair',Y,'nominal_frac_element1','actual_frac_element1']+columns].to_csv(OUT/'points.csv',index=False)
    scales={};global_limits={};local_limits={}
    for b in BASES:
        c='weighted_'+b;cols=[m+'_'+b for m in ['precursor','icp']]
        v=initial[cols].to_numpy();scales[c]=(v.min(),np.ptp(v))
        global_limits[c]=padded(data[cols].to_numpy());local_limits[c]=padded(bo[cols].to_numpy())
        joint.LABELS[c]=PROPERTIES[b][1]
    joint.configure_matplotlib()
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.family':'Arial','mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
    audit=[];probes={}
    def contour(ax,frame,x,y,xlim,ylim):
        assert frame.cohort.eq('initial').all()
        p=frame[[x,y]].to_numpy();origin=np.array([scales[c][0] for c in [x,y]]);span=np.array([scales[c][1] for c in [x,y]])
        model=KNeighborsRegressor(n_neighbors=36,weights='distance').fit((p-origin)/span,frame[Y])
        xx,yy=np.meshgrid(np.linspace(*xlim,220),np.linspace(*ylim,220));q=np.c_[xx.ravel(),yy.ravel()]
        values=model.predict((q-origin)/span);viewspan=np.array([np.ptp(xlim),np.ptp(ylim)])
        distances=NearestNeighbors(n_neighbors=1).fit(p/viewspan).kneighbors(q/viewspan)[0][:,0]
        shown=distances<=.055;z=np.ma.masked_where(~shown.reshape(xx.shape),values.reshape(xx.shape))
        cmap=ListedColormap(joint.BIN_COLORS);norm=BoundaryNorm(joint.BIN_EDGES,cmap.N,clip=True)
        ax.contourf(xx,yy,z,levels=joint.BIN_EDGES,cmap=cmap,norm=norm,alpha=.30,zorder=0)
        ax.contour(xx,yy,z,levels=joint.BIN_EDGES,colors='#777777',linewidths=.35,alpha=.30,zorder=1)
        key=(mode,x,y);probe=model.predict((p[:5]-origin)/span)
        if key in probes:np.testing.assert_allclose(probes[key],probe,rtol=0,atol=0)
        else:probes[key]=probe
        name=f'{mode}_{view}_{x}_{y}.npz'
        np.savez_compressed(OUT/name,x=xx,y=yy,mean=values.reshape(xx.shape),shown=shown.reshape(xx.shape))
        audit.append(dict(mode=mode,view=view,x=x,y=y,supported_pixels=int(shown.sum())))
    old=joint.draw_masked_knn_contour;joint.draw_masked_knn_contour=contour
    fig=plt.figure(figsize=(12.5,13.4))
    grid=fig.add_gridspec(4,3,left=.10,right=.99,bottom=.08,top=.925,hspace=.64,wspace=.42)
    try:
        for row,(mode,view) in enumerate([('precursor','global'),('icp','global'),('precursor','local'),('icp','local')]):
            frame=data.rename(columns={mode+'_'+b:'weighted_'+b for b in BASES})
            for col,(x,y) in enumerate(itertools.combinations(['weighted_'+b for b in BASES],2)):
                top,ax,side=joint.draw_joint_panel(fig,grid[row,col],frame.loc[frame.cohort.eq('initial')],frame.loc[frame.cohort.eq('BO')],x,y,global_limits if view=='global' else local_limits,mode='contour')
                style_joint(top,ax,chr(65+row*3+col)+'  '+('Precursor' if mode=='precursor' else 'ICP')+' | '+view.title(),dense=True)
        handles=[Patch(facecolor='none',edgecolor='none',label='H/D Separation Coefficient')]+[Patch(facecolor=c,edgecolor='black',label=l) for c,l in zip(joint.BIN_COLORS,joint.BIN_LABELS)]
        fig.legend(handles=handles,loc='upper center',ncol=6,bbox_to_anchor=(.5,.995),frameon=True,fancybox=False,edgecolor='black',fontsize=16,columnspacing=.8,handlelength=1.4)
        for ext in ['png','svg','pdf']:fig.savefig(OUT/('weighted_descriptor_maps.'+ext),dpi=200,facecolor='white')
    finally:
        plt.close(fig);joint.draw_masked_knn_contour=old
    inputs=[Path(__file__),HELPER,ROOT/'workflow/data.py',ROOT/'weighted_descriptor_helpers.py',ROOT/'plot_six_weighted_descriptor_pairs.py',*list((ROOT/'inputs/reference').glob('*')),*list((ROOT/'inputs/icp').glob('*.xlsx'))]
    manifest=dict(counts=data.cohort.value_counts().to_dict(),properties=BASES,colors=joint.BIN_COLORS,
        global_limits=global_limits,local_limits=local_limits,panels=audit,
        method='Matched complete properties across both representations and all three axes; no imputation; 36-neighbor distance-weighted initial H/D mean; fixed pooled initial range normalization; display-normalized proximity mask <=0.055; no hull mask; BO overlay only',
        local_policy='Pooled BO min/max plus 5.5% padding, no H/D-dependent selection',
        verification='Global/local predictions identical at fixed probes',
        hashes={p.relative_to(ROOT.parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},response_changed=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    label='Weighted descriptor maps'
    (OUT/'README.md').write_text('# Weighted descriptor maps\n\nRun `python plot_weighted_trio_letter.py`. The three columns show work function/H adsorption free energy, work function/electronegativity, and H adsorption free energy/electronegativity. Four rows show precursor/global, ICP/global, precursor/local, ICP/local.\n\nAll panels use the same complete-data cohort. Precursor weights are volume fractions under the equal-stock-concentration convention; ICP weights are metal molar fractions. Local bounds cover pooled BO ranges with 5.5% padding. Backgrounds use initial observations only; BO points are measured overlays. This script regenerates descriptive maps without fitting predictive ML models or editing Word.\n',encoding='utf-8')
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>'+label+'</title><style>body{font-family:Arial;margin:24px}img{width:100%;max-width:1800px}</style><h1>'+label+'</h1><p>Same complete-data cohort: '+str(len(initial))+' initial + '+str(len(bo))+' BO. Work function, H adsorption free energy and electronegativity; discrete colors and 12 panels.</p><a href="weighted_descriptor_maps.svg">Editable SVG</a> | <a href="weighted_descriptor_maps.pdf">PDF</a><img src="weighted_descriptor_maps.png">',encoding='utf-8')
    print(json.dumps(manifest['counts']))


if __name__=='__main__':main()
