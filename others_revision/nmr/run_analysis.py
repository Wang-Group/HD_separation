"""Reproduce adopted saved-phase NMR results and separate robustness checks."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import nmrglue as ng
import phase_validation as phase

INPUT = Path(__file__).resolve().parent
ROOT = INPUT/'results'
MAPPING = json.loads((INPUT/'sample_mapping.json').read_text(encoding='utf-8'))
CFG = json.loads((INPUT/'config.json').read_text(encoding='utf-8'))
PURITY = CFG['stock_d_atom_percent']
WIDTHS = [.15, .25, .4]
METHODS = ['saved_1r', 'ACME']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(name, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x))
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False)+'\n', encoding='utf-8')


def csvout(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def freeze():
    entries = json.loads((INPUT/'input_manifest.json').read_text(encoding='utf-8'))
    for e in entries:
        assert sha(INPUT/e['path']) == e['sha256'], e['path']
    return entries


def integrate(x, y, c, h):
    lo, hi = c-h, c+h
    lm = (x>=lo-.15)&(x<=lo-.05); rm = (x>=hi+.05)&(x<=hi+.15)
    assert lm.sum()>=5 and rm.sum()>=5
    xx = np.r_[lo, x[(x>lo)&(x<hi)], hi]
    base = np.interp(xx, [np.median(x[lm]),np.median(x[rm])], [np.median(y[lm]),np.median(y[rm])])
    yy = np.interp(xx,x,y)-base
    area = float(np.trapezoid(yy,xx))
    negative = float(np.trapezoid(np.maximum(-yy,0),xx)/np.trapezoid(abs(yy),xx))
    return area, negative


def process():
    rows, meta = [], []
    (ROOT/'processed_arrays').mkdir(exist_ok=True)
    for s in MAPPING['samples']:
        f = INPUT/'inputs'/s['id']
        dic, yr = ng.bruker.read_pdata(str(f/'pdata/1'))
        pr = dic['procs']; acq, fid = ng.bruker.read(str(f)); ac = acq['acqus']
        assert ac['NUC1']=='2H' and pr['SI']==65536 and pr['LB']==1 and pr['WDW']==1
        x = (pr['OFFSET']-np.arange(pr['SI'])*pr['SW_p']/(pr['SF']*pr['SI']))[::-1]
        dtype = ('<' if pr['BYTORDP']==0 else '>')+('i4' if pr['DTYPP']==0 else 'f8')
        yi = np.fromfile(f/'pdata/1/1i',dtype=dtype)*2.**pr['NC_proc']
        saved = (np.asarray(yr,float)+1j*yi)[::-1]
        corrected = ng.bruker.remove_digital_filter(acq,fid,truncate=False)
        z = ng.proc_base.fft_positive(ng.proc_base.zf_size(ng.proc_base.em(corrected,lb=1./ac['SW_h']),65536))[::-1]*2.**ac['NC']
        corr = float(np.corrcoef(abs(z),abs(saved))[0,1]); assert corr>.999
        ranges = [(-3.1,-2.2),(-1.3,.1)] if s['id']=='feed_dir7' else [(1.5,3.5),(3.6,6.5)]
        dc,wc = [phase.peak(x,abs(z),*bounds) for bounds in ranges]
        acme = phase.phase_and_measure(x,z,dc,wc,h=.25)
        fid_hash = sha(f/'fid')
        overflow = []
        if (f/'overflow_log_dru1.txt').exists():
            for line in (f/'overflow_log_dru1.txt').read_text(encoding='utf-8',errors='replace').splitlines():
                parts=line.split()
                if len(parts)==3 and all(p.isdigit() for p in parts): overflow.append([int(p) for p in parts])
        meta.append(dict(**s, fid_sha256=fid_hash, saved_real_sha256=sha(f/'pdata/1/1r'),
                         reference_center_original_ppm=dc, water_center_original_ppm=wc,
                         reference_alignment_delta_ppm=2.5-dc, water_aligned_ppm=wc+2.5-dc,
                         magnitude_correlation=corr, NS=ac['NS'], RG=ac['RG'], D1_s=ac['D'][1],
                         saved_PHC0=pr['PHC0'],saved_PHC1=pr['PHC1'],overflow_rows=overflow,
                         ACME_reference_phase_deg=acme['phase_ref_deg'],ACME_water_phase_deg=acme['phase_water_deg'],
                         ACME_search_boundary=acme['phase_search_boundary']))
        for method,y in [('saved_1r',saved.real),('ACME',acme['y'])]:
            for h in WIDTHS:
                ra,rn=integrate(x,y,dc,h); wa,wn=integrate(x,y,wc,h)
                assert ra>0 and wa>0
                rows.append(dict(**s, method=method, halfwidth_ppm=h, fid_sha256=fid_hash,
                                 reference_area=ra,water_area=wa,ratio=wa/ra,
                                 negative_fraction_reference=rn,negative_fraction_water=wn))
        np.savez_compressed(ROOT/'processed_arrays'/f"{s['id']}.npz",ppm=x,ppm_aligned=x+2.5-dc,
                            saved_real=saved.real,ACME_real=acme['y'],reference_center=dc,water_center=wc)
    dump('processing_records.json',meta);csvout('integrals.csv',rows)
    return rows,meta


def fit(rows,method,width=.25,exclude=()):
    select=[r for r in rows if r['role']=='standard' and r['method']==method and r['halfwidth_ppm']==width and r['id'][9:] not in exclude]
    grouped={g:[r for r in select if r['group']==g] for g in ['low','high']}
    fl=np.array([r['nominal_volume_percent']/100 for r in grouped['low']]);yl=np.array([r['ratio'] for r in grouped['low']])
    X=np.column_stack([np.ones(len(fl)),fl]);beta=np.linalg.lstsq(X,yl,rcond=None)[0]
    kl=beta.sum()/PURITY;x0=beta[0]/kl;rl=yl-X@beta
    fh=np.array([r['nominal_volume_percent']/100 for r in grouped['high']]);yh=np.array([r['ratio'] for r in grouped['high']])
    xh=x0+(PURITY-x0)*fh;kh=xh@yh/(xh@xh);rh=yh-kh*xh
    return dict(low=dict(k=kl,x0=x0,beta=beta,covbase=np.linalg.inv(X.T@X),rss=rl@rl,dof=len(fl)-2),
                high=dict(k=kh,x0=x0,rss=rh@rh,dof=len(fh)-1),high_f=fh,high_y=yh)


def propagate(model,measurements,draws=50000,seed=20261023):
    rng=np.random.default_rng(seed);lo=model['low']
    vl=lo['rss']/rng.chisquare(lo['dof'],draws)
    beta=lo['beta']+(rng.normal(size=(draws,2))@np.linalg.cholesky(lo['covbase']).T)*np.sqrt(vl)[:,None]
    kl=beta.sum(axis=1)/PURITY;x0=beta[:,0]/kl
    xh=x0[:,None]+(PURITY-x0[:,None])*model['high_f'];xx=(xh*xh).sum(axis=1)
    khc=xh@model['high_y']/xx
    vh=((model['high_y']-khc[:,None]*xh)**2).sum(axis=1)/rng.chisquare(model['high']['dof'],draws)
    kh=khc+rng.normal(size=draws)*np.sqrt(vh/xx)
    assert np.all(kl>0) and np.all(kh>0)
    gains={'low':kl,'high':kh};variance={'low':vl,'high':vh}
    cal,reading,noise={},{},{}
    for r in measurements:
        key=r['fid_sha256']
        if key not in noise:noise[key]=rng.normal(size=draws)*np.sqrt(variance[r['group']])
        cal[r['id']]=r['ratio']/gains[r['group']]
        reading[r['id']]=(r['ratio']+noise[key])/gains[r['group']]
    return dict(gains=gains,x0=x0,calibration=cal,with_reading=reading)


def bounds(a):return [float(v) for v in np.quantile(a,[.025,.975])]


def calculate(rows,meta):
    main={m:fit(rows,m) for m in METHODS}
    measured={m:[r for r in rows if r['role']!='standard' and r['method']==m and r['halfwidth_ppm']==.25] for m in METHODS}
    keyed={m:{r['id']:r for r in v} for m,v in measured.items()}
    mc={m:propagate(main[m],measured[m]) for m in METHODS}
    concentrations=[]
    for r in measured['saved_1r']:
        ident=r['id'];out={k:r[k] for k in ['id','role','group','stage','feed_directory_batch','fid_sha256']}
        out['feed_basis']=r.get('feed_basis')
        for m in METHODS:
            rr=keyed[m][ident];d=rr['ratio']/main[m][r['group']]['k']
            out[m+'_ratio']=rr['ratio'];out[m+'_D_percent']=d
            out[m+'_calibration_95_low'],out[m+'_calibration_95_high']=bounds(mc[m]['calibration'][ident])
            out[m+'_model_reading_95_low'],out[m+'_model_reading_95_high']=bounds(mc[m]['with_reading'][ident])
            out[m+'_draws_above100']=int((mc[m]['with_reading'][ident]>100).sum())
            out[m+'_draws_below0']=int((mc[m]['with_reading'][ident]<0).sum())
        out['saved_vs_uniform_ACME_relative_percent']=100*(out['saved_1r_D_percent']/out['ACME_D_percent']-1)
        concentrations.append(out)
    cbyid={r['id']:r for r in concentrations}
    pairs=[];checks=[]
    for stage in range(1,8):
        product=f'product_{stage}';feeds=[r for r in measured['saved_1r'] if r['role']=='feed' and r['stage']==stage]
        out={'stage':stage,'product_id':product,'feed_id':None,'status':'actual_feed_not_identified'}
        for m in METHODS:out[m+'_product_D_percent']=cbyid[product][m+'_D_percent']
        if feeds:
            feed=feeds[0]['id'];out.update(feed_id=feed,status='user_confirmed_pair_and_preparation',feed_directory_batch=feeds[0]['feed_directory_batch'])
            out['feed_basis']=feeds[0].get('feed_basis','prepared feed NMR, user-confirmed pairing and preparation')
            if feed=='feed_stage2_proxy':out['status']='user_confirmed_direct_transfer_reuses_product1_NMR'
            for m in METHODS:
                out[m+'_feed_D_percent']=cbyid[feed][m+'_D_percent']
                e=cbyid[product][m+'_D_percent']/cbyid[feed][m+'_D_percent'];out[m+'_E']=e
                ce=mc[m]['calibration'][product]/mc[m]['calibration'][feed]
                re=mc[m]['with_reading'][product]/mc[m]['with_reading'][feed]
                assert np.isfinite(re).all()
                out[m+'_E_feed_nonpositive_draws']=int((mc[m]['with_reading'][feed]<=0).sum())
                out[m+'_E_product_nonpositive_draws']=int((mc[m]['with_reading'][product]<=0).sum())
                out[m+'_E_calibration_95_low'],out[m+'_E_calibration_95_high']=bounds(ce)
                out[m+'_E_model_reading_95_low'],out[m+'_E_model_reading_95_high']=bounds(re)
                rp=keyed[m][product];rf=keyed[m][feed]
                if rp['group']==rf['group']:
                    error=float(np.max(abs(ce-e)));assert error<1e-12
                    checks.append(dict(check='same_group_calibration_cancels',stage=stage,method=m,max_error=error))
                else:
                    expect=(rp['ratio']/rf['ratio'])*mc[m]['gains'][rf['group']]/mc[m]['gains'][rp['group']]
                    assert np.allclose(ce,expect,rtol=1e-14)
            out['E_absolute_change']=out['saved_1r_E']-out['ACME_E']
            out['E_relative_change_percent']=100*(out['saved_1r_E']/out['ACME_E']-1)
        pairs.append(out)
    sensitivities=[]
    for method in METHODS:
        for width in WIDTHS:
            for scenario,exclude in [('all_labels',()),('exclude_overflow_0p3',('0.3',)),('exclude_unverified_50',('50',)),('exclude_both',('0.3','50'))]:
                model=fit(rows,method,width,exclude)
                selected={r['id']:r for r in rows if r['method']==method and r['halfwidth_ppm']==width and r['role']!='standard'}
                for pair in pairs:
                    if pair['feed_id'] is None:continue
                    p,f=selected[pair['product_id']],selected[pair['feed_id']]
                    pd=p['ratio']/model[p['group']]['k'];fd=f['ratio']/model[f['group']]['k']
                    sensitivities.append(dict(method=method,halfwidth_ppm=width,scenario=scenario,stage=pair['stage'],feed_D_percent=fd,product_D_percent=pd,E=pd/fd))
    for m in METHODS:
        for r in measured[m]:
            if r['role']!='comparator':continue
            prod=keyed[m][f"product_{r['stage']}"]
            assert r['fid_sha256']==prod['fid_sha256']
            diff=abs(r['ratio']-prod['ratio']);assert diff<1e-12
            checks.append(dict(check='identical_FID_reused_role_integral',sample=r['id'],method=m,absolute_ratio_difference=diff))
    # Independent frozen standard integrals test both implementations.
    benchmark=json.loads((INPUT/'validation/standard_integrals_reference.json').read_text(encoding='utf-8'))
    previous={(r['sample'],r['method'],r['halfwidth_ppm']):r['ratio'] for r in benchmark['integrals']}
    standard_errors=[]
    for r in rows:
        if r['role']=='standard':
            diff=abs(r['ratio']-previous[(r['id'][9:],r['method'],r['halfwidth_ppm'])]);standard_errors.append(diff);assert diff<1e-10
    checks.append(dict(check='standard_method_selection_reproduction',ratios_checked=len(standard_errors),max_difference=max(standard_errors)))
    result=dict(methods=METHODS,calibrations=main,concentrations=concentrations,enrichment=pairs,
                sensitivity=sensitivities,checks=checks,
                definitions={'sample':'approximate atom%D = R/k; no blank subtraction, no additional0.95 correction',
                             'enrichment':'E = product atom%D / measured feed atom%D; not gas/liquid separation coefficient',
                             'ACME_comparator':'Robustness check only; response figures and tables use saved_1r.',
                             'uncertainty':'50000 shared sequential Student-t calibration draws; reading intervals additionally assume residual-based independent errors once per uniqueFID. Preparation/systematic phase/stock purity uncertainty excluded.',
                             'standard_truth':'nominal stock volume labels with prior approximate volume-to-atom mixing convention',
                             'stage1_2':'User confirmed independently measured pure-water record represents stage1 feed; stage2 directly receives product1 without dilution and reuses its NMR, not an independent measurement.'},
                qc=meta)
    dump('migration_results.json',result)
    csvout('sample_concentrations.csv',concentrations);csvout('enrichment_comparison.csv',pairs)
    csvout('window_and_standard_sensitivity.csv',sensitivities)
    dump('verification.json',checks)
    np.savez_compressed(ROOT/'calibration_draws.npz',**{m+'_'+k:v for m in METHODS for k,v in mc[m]['gains'].items()})
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.output.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest=freeze();rows,meta=process();result=calculate(rows,meta)
    assert all(sha(INPUT/e['path'])==e['sha256'] for e in manifest)
    print(json.dumps({'status':'complete','role_spectra':len(meta),'unique_fids':len({r['fid_sha256'] for r in meta}),
                      'enrichment':[{k:v for k,v in r.items() if k in ['stage','saved_1r_feed_D_percent','saved_1r_product_D_percent','saved_1r_E','ACME_E','E_relative_change_percent']} for r in result['enrichment']]},ensure_ascii=False))
