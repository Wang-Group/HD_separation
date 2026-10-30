"""Combine the saved calibration spectra for the SI without refitting calibration."""
import csv
from pathlib import Path
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT/'results'
OUT = DATA/'figures'


def peak_region(x, y, center):
    left = (x >= center-.40) & (x <= center-.30)
    right = (x >= center+.30) & (x <= center+.40)
    bx = np.array([np.median(x[left]), np.median(x[right])])
    by = np.array([np.median(y[left]), np.median(y[right])])
    xx = np.r_[center-.25, x[(x > center-.25) & (x < center+.25)], center+.25]
    yy = np.interp(xx, x, y)-np.interp(xx, bx, by)
    return xx, yy, bx, by


def main():
    with (DATA/'integrals.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['role']=='standard'
                and r['method']=='saved_1r' and float(r['halfwidth_ppm'])==.25]
    assert len(rows)==16 and len({r['id'] for r in rows})==16
    plt.rcParams.update({'font.family':'Arial', 'font.size':9, 'svg.fonttype':'none',
                         'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False})
    for group, count in [('low',11), ('high',5)]:
        selected = sorted([r for r in rows if r['group']==group],
                          key=lambda r:(float(r['nominal_volume_percent']), r['id']))
        assert len(selected)==count
        nrows = (count+1)//2
        fig, axes = plt.subplots(nrows,2,figsize=(7,8.1 if group=='low' else 4.9),sharex=True)
        fig.subplots_adjust(left=.08,right=.98,bottom=.08 if group=='low' else .13,
                            top=.94 if group=='low' else .88,hspace=.45,wspace=.20)
        fig.suptitle(('Low' if group=='low' else 'High')+'-response calibration spectra',fontsize=12)
        for i,(ax,row) in enumerate(zip(axes.flat,selected)):
            z = np.load(DATA/'processed_arrays'/f"{row['id']}.npz")
            x,y = z['ppm'],z['saved_real']
            water,ref = float(z['water_center']),float(z['reference_center'])
            regions = [peak_region(x,y,water), peak_region(x,y,ref)]
            areas = [np.trapezoid(r[1],r[0]) for r in regions]
            for area, key in zip(areas,['water_area','reference_area']):
                assert np.isclose(area,float(row[key]),rtol=2e-12,atol=1e-7), row['id']
            bx=np.concatenate([r[2] for r in regions]); by=np.concatenate([r[3] for r in regions])
            order=np.argsort(bx); corrected=y-np.interp(x,bx[order],by[order])
            shift=2.5-ref; mask=(x+shift>=1)&(x+shift<=6)
            ax.plot((x+shift)[mask],corrected[mask]/max(corrected[mask]),color='#385468',lw=.75)
            for center,color in [(water,'#138f7b'),(ref,'#b7842c')]:
                ax.axvspan(center+shift-.25,center+shift+.25,color=color,alpha=.10)
            nominal=float(row['nominal_volume_percent'])
            label=f'{nominal:g}% stock' if nominal else ('Water 1' if row['id']=='standard_H2O' else 'Water 2')
            ax.set_title(f'{string.ascii_uppercase[i]}  {label}',loc='left',fontsize=9.5,pad=3)
            ax.text(.98,.86,f"Water/DMSO = {float(row['ratio']):.3f}",transform=ax.transAxes,
                    ha='right',fontsize=7.5)
            ax.set_xlim(6,1); ax.set_ylim(-.10,1.35); ax.set_yticks([])
            ax.spines['left'].set_visible(False); ax.axhline(0,color='#899399',lw=.4)
            ax.tick_params(axis='x',labelbottom=True,labelsize=8)
        for ax in list(axes.flat)[count:]:
            ax.set_visible(False)
        fig.supxlabel('Chemical shift aligned to DMSO at 2.50 ppm',fontsize=10,y=.015)
        fig.supylabel('Intensity (independently normalized)',fontsize=9,x=.012)
        for extension in ('png','pdf','svg'):
            fig.savefig(OUT/f'nmr_calibration_spectra_{group}.{extension}',dpi=240,facecolor='white')
        plt.close(fig)
        print(f'{group}: {count} saved spectra; all peak areas match saved integrals')


if __name__=='__main__':
    main()
