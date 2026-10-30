"""Reproduce NMR calibration, enrichment and spectra from the saved-phase calculation."""
from pathlib import Path
import argparse
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, default=Path(__file__).parent/'results')
args = parser.parse_args()
DATA = args.results.resolve()
OUT = DATA/'figures'
OUT.mkdir(parents=True, exist_ok=True)
d = json.loads((DATA/'migration_results.json').read_text(encoding='utf-8'))
rows = list(csv.DictReader((DATA/'integrals.csv').open(encoding='utf-8-sig', newline='')))
ratios = {r['id']: r for r in rows if r['method']=='saved_1r' and float(r['halfwidth_ppm'])==.25}
products = sorted([r for r in d['concentrations'] if r['role']=='product'], key=lambda r:r['stage'])
enrich = sorted(d['enrichment'], key=lambda r:r['stage'])
assert len(products)==len(enrich)==7
cal = d['calibrations']['saved_1r']
plt.rcParams.update({'font.family':'Arial', 'font.size':10, 'axes.spines.top':False,
    'axes.spines.right':False, 'pdf.fonttype':42, 'svg.fonttype':'none',
    'mathtext.fontset':'custom', 'mathtext.rm':'Arial', 'mathtext.it':'Arial:italic',
    'mathtext.bf':'Arial:bold', 'figure.facecolor':'white'})

def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=240, facecolor='white')
    plt.close(fig)

# NMR calibration: the same sequential two-group calibration, with current integrals.
fig,ax=plt.subplots(2,2,figsize=(9,6.05),gridspec_kw={'height_ratios':[3,1]})
fig.subplots_adjust(left=.105,right=.975,top=.90,bottom=.115,hspace=.13,wspace=.29)
fig.suptitle('Reintegrated $^2$H NMR calibration',fontsize=15)
r2={}
for j,group in enumerate(['low','high']):
    rr=[r for r in ratios.values() if r['role']=='standard' and r['group']==group]
    f=np.array([float(r['nominal_volume_percent'])/100 for r in rr])
    x=cal['low']['x0']+(99.86-cal['low']['x0'])*f
    y=np.array([float(r['ratio']) for r in rr]); pred=cal[group]['k']*x
    r2[group]=1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
    xmax=.53 if group=='low' else 103
    ax[0,j].plot([0,xmax],[0,cal[group]['k']*xmax],color='#315d78',lw=1.3)
    ax[0,j].scatter(x,y,s=32,facecolors='white',edgecolors='#315d78',zorder=3)
    ax[0,j].set_title(('A  Low-range group' if j==0 else 'B  High-range group'),loc='left',fontsize=11)
    ax[0,j].text(.05,.88,f"k = {cal[group]['k']:.6f}\n$R^2$ = {r2[group]:.6f}",transform=ax[0,j].transAxes,fontsize=10)
    ax[0,j].set_ylabel('Water / DMSO integral ratio')
    ax[0,j].tick_params(labelbottom=False)
    ax[1,j].axhline(0,color='#777777',lw=.8)
    ax[1,j].scatter(x,y-pred,s=25,color='#315d78')
    ax[1,j].set_ylabel('Residual');ax[1,j].set_xlabel('Calibration input (atom% D)')
    for a in ax[:,j]: a.set_xlim(-.012 if j==0 else -2,xmax)
save(fig,'nmr_calibration')

# stagewise deuterium enrichment: product concentrations only; preserve the original two-panel scope.
fig,ax=plt.subplots(1,2,figsize=(10.8,4.0))
fig.subplots_adjust(left=.085,right=.98,top=.87,bottom=.18,wspace=.28)
ys=[p['saved_1r_D_percent'] for p in products]
for j,a in enumerate(ax):
    a.plot(range(1,8),ys,'o-',color='#315d78',lw=1.5,ms=5)
    a.set_xticks(range(1,8));a.set_xlabel('Enrichment stage');a.set_ylabel('Product concentration (atom% D)')
    a.set_title(('A  Linear scale' if j==0 else 'B  Logarithmic scale'),loc='left',fontsize=11)
    a.set_xlim(.65,7.35);a.grid(axis='y',alpha=.18)
ax[0].set_ylim(0,112);ax[1].set_yscale('log');ax[1].set_ylim(.045,180)
for stage,y in enumerate(ys,1):
    ax[1].annotate(f'{y:.4f}' if y<1 else f'{y:.2f}',(stage,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
save(fig,'stagewise_deuterium_enrichment')

def region(x,y,c):
    h=.25
    lm=(x>=c-h-.15)&(x<=c-h-.05);rm=(x>=c+h+.05)&(x<=c+h+.15)
    bx=np.array([np.median(x[lm]),np.median(x[rm])]);by=np.array([np.median(y[lm]),np.median(y[rm])])
    xx=np.r_[c-h,x[(x>c-h)&(x<c+h)],c+h]
    yy=np.interp(xx,x,y)-np.interp(xx,bx,by)
    return xx,yy,bx,by

# stage-product NMR spectra: saved real spectra, aligned reference and signed cumulative integrals.
fig,axes=plt.subplots(7,1,figsize=(8.3,9.3),sharex=True)
fig.subplots_adjust(left=.075,right=.99,top=.945,bottom=.055,hspace=.13)
fig.suptitle('Stage-product $^2$H NMR spectra and peak integrals',fontsize=13,y=.985)
area_checks=[]
for a,p in zip(axes,products):
    z=np.load(DATA/'processed_arrays'/f"{p['id']}.npz")
    x=z['ppm'];y=z['saved_real'];ref=float(z['reference_center']);water=float(z['water_center']);shift=2.5-ref
    reg=[region(x,y,water),region(x,y,ref)]
    bx=np.concatenate([r[2] for r in reg]);by=np.concatenate([r[3] for r in reg]);order=np.argsort(bx)
    yc=y-np.interp(x,bx[order],by[order]);mask=(x+shift>=1)&(x+shift<=6)
    scale=max(yc[mask]);a.plot((x+shift)[mask],yc[mask]/scale,color='#385468',lw=.8)
    vals=[float(np.trapezoid(r[1],r[0])) for r in reg]
    for value,key in zip(vals,['water_area','reference_area']):
        expected=float(ratios[p['id']][key]);assert np.isclose(value,expected,rtol=2e-12,atol=1e-7)
        area_checks.append({'id':p['id'],'peak':key,'absolute_error':abs(value-expected)})
    for (xx,yy,_,_),color in zip(reg,['#138f7b','#b7842c']):
        a.axvspan(xx[0]+shift,xx[-1]+shift,color=color,alpha=.10)
        cum=-cumulative_trapezoid(yy[::-1],xx[::-1],initial=0)
        a.plot(xx[::-1]+shift,1.05+.35*cum/max(vals),color=color,lw=1.2)
    a.text(.012,.88,f"Stage {p['stage']}  |  {p['saved_1r_D_percent']:.4f} atom% D",transform=a.transAxes,fontsize=9,fontweight='bold')
    a.text(.99,.88,f"Water / DMSO = {p['saved_1r_ratio']:.4f}",transform=a.transAxes,ha='right',fontsize=8.5)
    a.set_ylim(-.10,1.72);a.set_yticks([]);a.spines['left'].set_visible(False);a.axhline(0,color='#899399',lw=.4)
axes[-1].set_xlim(6,1);axes[-1].set_xlabel('Chemical shift aligned to DMSO at 2.50 ppm')
fig.text(.018,.5,'Intensity (independently normalized for display)',rotation=90,va='center',fontsize=9)
save(fig,'stage_product_nmr_spectra')

