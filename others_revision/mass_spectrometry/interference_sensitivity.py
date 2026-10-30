"""Conditional H3 sensitivity from first-hand H2-only controls, not a correction."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/interference_sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
RAW = ROOT / 'raw/mass_calibration/hiden_data/file147.csv'
# Rounded equations transcribed from the calibration figure supplied by the user.
A2, B2, A3, B3 = 2.06e-7, -1.31e-9, 6.16e-8, 9.51e-12
LIQUID_HD = 9.0  # 10 atom% D; D2 in the gas is neglected.
data = pd.read_csv(RAW, skiprows=24)
t, h2, m3 = (data.iloc[:, i].to_numpy(float) for i in (1,2,3))
t /= 3600000
flag = (h2 > 1e-8) & (t >= 1.2) & (t <= 14)
starts = np.where(np.diff(flag.astype(int), prepend=0)==1)[0]
ends = np.where(np.diff(flag.astype(int), append=0)==-1)[0]
records = []
for a,b in zip(starts, ends):
    if b-a < 10:
        continue
    core = (t >= t[a]+.25*(t[b]-t[a])) & (t <= t[a]+.75*(t[b]-t[a]))
    base = (t >= t[a]-.06) & (t <= t[a]-.02)
    records.append(dict(pulse=len(records)+1, start_h=t[a], end_h=t[b],
                        h2=float(np.median(h2[core])),
                        raw_h3=float(np.median(m3[core])),
                        net_h3=float(np.median(m3[core])-np.median(m3[base]))))
p = pd.DataFrame(records)
assert len(p)==40
previous = pd.read_csv(ROOT/'outputs/hiden_pulse_windows.csv')
assert np.allclose(p.h2, previous.h2_median_mbar, rtol=1e-10, atol=0)
assert np.allclose(p.net_h3, previous.net_m3_mbar, rtol=1e-10, atol=0)
p['x_h2'] = (p.h2-B2)/A2
p['in_representative_range'] = p.h2.between(4e-8,8e-8)
p.to_csv(OUT/'matched_controls.csv',index=False)

def apparent(alpha, x2, extra):
    x3 = 2*x2/(LIQUID_HD*alpha-1)
    return (2*x2/(x3+extra/A3)+1)/LIQUID_HD

rows=[]
for alpha in [2,5,10,15,20]:
    for _,r in p.iterrows():
        expected = A3*2*r.x_h2/(LIQUID_HD*alpha-1)
        for scenario in ['net_h3','raw_h3']:
            app=apparent(alpha,r.x_h2,r[scenario])
            rows.append(dict(pulse=int(r.pulse),h2=r.h2,true_alpha=alpha,scenario=scenario,
                             expected_hd_above_intercept=expected,
                             expected_m3_without_added_h3=expected+B3,
                             added_h3=r[scenario],apparent_alpha=app,
                             downward_bias_percent=100*(alpha-app)/alpha,
                             in_representative_range=bool(r.in_representative_range)))
results=pd.DataFrame(rows)
results.to_csv(OUT/'sensitivity_by_control.csv',index=False)
summary=[]
for subset,label in [(results,'all observed controls'),
                     (results[results.in_representative_range],'H2 within 4e-8 to 8e-8')]:
    for (alpha,scenario),d in subset.groupby(['true_alpha','scenario']):
        summary.append(dict(subset=label,true_alpha=int(alpha),scenario=scenario,n=len(d),
                            bias_min=d.downward_bias_percent.min(),
                            bias_median=d.downward_bias_percent.median(),
                            bias_max=d.downward_bias_percent.max(),
                            apparent_min=d.apparent_alpha.min(),apparent_max=d.apparent_alpha.max()))
pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)

plt.rcParams.update({'font.family':'Arial','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
ax=axes[0]
ax.scatter(p.h2/1e-8,p.raw_h3/1e-11,facecolors='none',edgecolors='gray',s=32,label='Raw control')
ax.scatter(p.h2/1e-8,p.net_h3/1e-11,color='#247a72',s=22,label='Background-subtracted')
ax.set(xlabel=r'H$_2$ signal ($10^{-8}$ mbar)',ylabel=r'Control m/z = 3 ($10^{-11}$ mbar)',title='A  Matched H$_2$-only controls')
ax.legend(fontsize=9)
colors={2:'#65753c',5:'#7e6a9a',10:'#287ba0',15:'#b75b24',20:'#a53860'}
ax=axes[1]
for alpha,color in colors.items():
    d=results[(results.true_alpha==alpha)&(results.scenario=='net_h3')]
    ax.scatter(d.h2/1e-8,d.downward_bias_percent,s=20,color=color,label=fr'$\alpha_{{true}}$ = {alpha}')
ax.set(xlabel=r'H$_2$ signal ($10^{-8}$ mbar)',ylabel='Downward bias in separation coefficient (%)',
       title='B  Bias at the same H$_2$ signal',ylim=(0,None))
ax.legend(fontsize=9,ncol=2)
ax=axes[2]
x=np.linspace(2,20,201)
q=p
matrix=np.array([apparent(a,q.x_h2.to_numpy(),q.net_h3.to_numpy()) for a in x])
ax.plot(x,x,color='black',ls='--',lw=1,label='No added interference')
ax.fill_between(x,matrix.min(axis=1),matrix.max(axis=1),color='#247a72',alpha=.22,label='Observed control range')
ax.plot(x,np.median(matrix,axis=1),color='#247a72',lw=2,label='Median across controls')
ax.set(xlabel='True separation coefficient',ylabel='Apparent separation coefficient',
       title='C  Effect on the reported value',xlim=(2,20),ylim=(1.8,20.5))
ax.legend(fontsize=9)
for a in axes:
    a.grid(alpha=.15)
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'h3_separation_sensitivity.{ext}',dpi=240)
plt.close(fig)
manifest=dict(raw_source=RAW.relative_to(ROOT).as_posix(),raw_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
              calibration={'source':'User-supplied calibration figure, rounded displayed coefficients',
                           'H2_slope':A2,'H2_intercept':B2,'HD_slope':A3,'HD_intercept':B3},
              assumptions=['10 atom% D in liquid; gas D2 neglected.',
                           'Added m/z=3 interference is treated as residual beyond the calibration.',
                           'Control and calibration responses assumed transferable for this sensitivity calculation.',
                           'No extrapolation beyond observed H2 controls; 40 pulses, not 40 independent replicates.',
                           'Control spread is descriptive, not a confidence interval or experimental error bar.',
                           'Mixed-gas calibration may already absorb some H3 contribution; do not apply as a sample correction.'],
              primary_scenario='Local-background-subtracted control signal',
              alternative='Entire raw m/z=3 control signal assigned as additional interference',
              summary=summary)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
assert np.allclose(apparent(15,p.x_h2,0),15)
assert (results.apparent_alpha < results.true_alpha).all()
print(pd.DataFrame(summary).to_string(index=False))
