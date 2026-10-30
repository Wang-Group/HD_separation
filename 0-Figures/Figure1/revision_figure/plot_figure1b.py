"""Render panel B only from the reviewed LSV extraction and original HD data."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lsv_plot_data import load_data, original_hd, input_hashes

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'figure1b_inputs'
OUT = HERE / 'figure1b_outputs'
LOW, HIGH = 10., 550.
PALETTE = ['#2ca9df','#f3b61f','#6f2dbd','#d64f1a','#e4ce20','#db7eb3',
           '#7e8ce8','#9b5d16','#2bd8c4','#3447b5','#b3b66d','#df70b8','#43cba5','#8ecf6b']


def frontier(frame):
    result, best = [], -np.inf
    for index, row in frame.sort_values(['x','y'], ascending=[True,False]).iterrows():
        if row.y > best:
            result.append(index)
            best = row.y
    return frame.loc[result]


def main():
    OUT.mkdir(exist_ok=True)
    curves, stats = load_data()
    hd = original_hd()
    data = stats.merge(hd, on='sample', how='left', validate='one_to_one', indicator=True)
    assert data['_merge'].eq('both').all(), data.loc[data['_merge']!='both','sample'].tolist()
    data['y'] = data.original_HD_coefficient
    data['x'] = data.potential_mean_mV
    data['sd'] = data.potential_SD_mV
    data['metric'] = data.potential_metric
    left_only = data.metric.eq('no_accepted_result') & data.n_above_target_no_crossing.gt(0)
    data.loc[left_only,'metric'] = 'left_boundary_placeholder'
    drawable = data.y.notna() & (data.x.notna() | left_only)
    excluded = data[~drawable].copy()
    data = data[drawable].copy()
    data['display_x'] = data.x.clip(LOW,HIGH)
    data.loc[data.metric.eq('left_boundary_placeholder'),'display_x'] = LOW
    data['display_sd_low'] = (data.x-data.sd).clip(LOW,HIGH)
    data['display_sd_high'] = (data.x+data.sd).clip(LOW,HIGH)
    data['display_boundary'] = np.where(data.x < LOW,'below',np.where(data.x > HIGH,'above','none'))
    data.loc[data.metric.eq('left_boundary_placeholder'),'display_boundary'] = 'left_placeholder_not_eta10'
    lit = pd.read_csv(INPUT/'literature.csv',dtype={'source_label':str,'reference_label':str})
    lit['x'] = lit.plot_x_overpotential_mV
    lit['y'] = lit.separation_coefficient
    lit['display_x'] = lit.x.clip(LOW,HIGH)
    currents = lit.measurement_condition.fillna('').str.extractall(r'(\d+(?:\.\d+)?)\s*mA')[0].astype(float).groupby(level=0).max()
    lit['front_condition'] = currents.reindex(lit.index).ge(50).fillna(False)
    lit['eligible'] = lit.front_condition & ~(lit.source_label.eq('33') & lit.x.eq(550) & lit.y.eq(16.5))
    literature_front = frontier(lit[lit.eligible])
    eligible = data[data.metric.eq('eta10') & ~data.is_pure_metal]
    combined = pd.concat([eligible[['sample','x','y']],lit.loc[lit.eligible,['x','y']]],ignore_index=True)
    combined_front = frontier(combined)
    stars = combined_front[combined_front['sample'].notna()]
    star_names = set(stars['sample'])
    data['on_frontier'] = data['sample'].isin(star_names)
    data.to_csv(OUT/'plotted_samples.csv',index=False)
    excluded.to_csv(OUT/'unplotted_samples.csv',index=False)
    combined_front.to_csv(OUT/'combined_frontier.csv',index=False)
    lit.to_csv(OUT/'literature_points.csv',index=False)
    plt.rcParams.update({'font.family':'Arial','font.size':13,'axes.linewidth':1.3,
                         'mathtext.fontset':'custom','mathtext.rm':'Arial',
                         'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold',
                         'svg.fonttype':'none','pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(16,8))
    for f,color in [(literature_front,'#2459f5'),(combined_front,'#fa3538')]:
        x=f.x.clip(LOW,HIGH).to_numpy();y=f.y.to_numpy()
        ax.fill_between(x,2.3,y,color=color,alpha=.045,zorder=0)
        ax.plot(x,y,color=color,lw=2,zorder=2)
    for _,r in data.iterrows():
        x=r.display_x
        if pd.notna(r.sd):
            err=np.array([[x-r.display_sd_low],[r.display_sd_high-x]])
            ax.errorbar(x,r.y,xerr=err,fmt='none',ecolor='#909090',elinewidth=.7,alpha=.38,capsize=1.5,zorder=1)
        marker='<' if r.metric=='left_boundary_placeholder' else 'd' if r.metric=='estimate' else '>' if r.metric=='scan_limit' or r.x>HIGH else '<' if r.x<LOW else 'o'
        color='#b37c25' if r.metric=='estimate' else '#969696'
        ax.scatter(x,r.y,marker=marker,s=42 if marker=='o' else 66,facecolor=color if marker in ('o','d') else 'none',edgecolor='#555555',lw=.55,alpha=.65,zorder=3)
    names=sorted(lit.source_label.unique())
    colors={name:PALETTE[i%len(PALETTE)] for i,name in enumerate(names)}
    for name in names:
        for flag,marker in [(True,'^'),(False,'s')]:
            subset=lit[lit.source_label.eq(name) & lit.front_condition.eq(flag)]
            ax.scatter(subset.display_x,subset.y,s=75,marker=marker,color=colors[name],edgecolor='#444444',linewidth=.7,zorder=4,alpha=.9)
    ax.scatter(stars.x.clip(LOW,HIGH),stars.y,marker='*',s=210,color='#ff3538',edgecolor='black',linewidth=.7,zorder=7)
    handles=[Line2D([0],[0],color=colors[n],lw=4,label=n) for n in names]
    handles += [Line2D([0],[0],marker=m,color='none',markeredgecolor='#444444',markerfacecolor=c,markersize=s,label=t)
                for m,c,s,t in [('^','white',8,r'Paper $\geq$50 mA cm$^{-2}$'),('s','white',8,'Paper <50/unclear'),
                              ('o','#999999',7,r'Ours: mean $\pm$ SD'),('*','#ff3538',12,'Ours: frontier'),
                              ('d','#b37c25',7,r'Estimate (not $\eta_{10}$)'),('>','white',7,'Scan/display bound'),
                              ('<','white',7,'No accepted result')]]
    legend=ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.008,1.,.984,0),
                     mode='expand',ncol=12,fontsize=11.5,handlelength=.85,
                     columnspacing=.65,handletextpad=.35,borderpad=.35,
                     labelspacing=.45,borderaxespad=0,framealpha=.95)
    fig.text(.024,.99,'B',fontsize=25,fontweight='bold',va='top')
    ax.set(xlim=(0,560),ylim=(2.3,16))
    ax.set_xlabel(r'Overpotential at -10 mA cm$^{-2}$ (mV)',fontsize=19,fontweight='bold')
    ax.set_ylabel(r'H/D separation coefficient',fontsize=19,fontweight='bold')
    ax.tick_params(labelsize=16)
    fig.subplots_adjust(left=.095,right=.985,bottom=.14,top=.95)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    legend_box=legend.get_window_extent(renderer)
    assert all(legend_box.contains(*text.get_window_extent(renderer).get_points()[0]) and
               legend_box.contains(*text.get_window_extent(renderer).get_points()[1])
               for text in legend.get_texts()), 'Legend text exceeds frame'
    assert legend_box.y0 > ax.transData.transform((HIGH,14.5))[1] + 7, 'Legend obscures highest literature point'
    for suffix in ['png','pdf','svg']:
        fig.savefig(OUT/('Figure1B-revision.'+suffix),dpi=250,facecolor='white')
    plt.close(fig)
    info=dict(plotted_samples=len(data),excluded_samples=len(excluded),
              metrics=data.metric.value_counts().to_dict(),frontier_samples=stars['sample'].tolist(),
              HD_policy='Original HD values only; no newly added HD measurements or HD error bars.',
              LSV_policy='Crossing eta10 mean and sample SD. If no crossings, average adopted rising/nearest estimates together, explicitly as estimates. Otherwise show furthest scan limit or no-result placeholder.',
              grouping_policy='Pool audited same-recipe individual curves once under ordinary catalyst names. Initial-submission HD means are pooled by their measurement counts.',
              no_crossing_policy='Samples without a usable result shown at 10 mV as open left triangles; display placeholders only. No assigned eta10 or SD.',
              frontier_policy='Unclipped eta10 means; rising points, bounds and pure metals excluded. Same-recipe groups treated as ordinary samples.',
              display_window=[LOW,HIGH],axis_window=[0,560],
              figure_size_inches=[16,8], catalyst_annotations=False,
              literature_identity_note='Catalyst labels omitted for manual annotation in Inkscape.',
              input_sha256={**input_hashes(), **{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), *INPUT.glob('*.csv')]}})
    (OUT/'provenance.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    (OUT/'caption.txt').write_text('Figure 1B. Original H/D separation coefficients versus reviewed LSV overpotential. '
        'Horizontal bars show mean +/- sample SD of accepted eta10 measurements (n >= 2). '
        'Diamonds denote mean rising/nearest-point estimates for samples without accepted eta10 and are not measured eta10; their bars, where present, are estimate SDs. '
        'Scan-only samples are shown at their furthest tested limit, not as averaged eta10. '
        'Open left triangles at 10 mV represent samples with no usable result whose recorded current magnitude remains above 10 mA/cm2; '
        'this is a display placeholder, not measured eta10 or evidence of high HER activity. These samples are excluded from the Pareto front. '
        'Means and uncertainty endpoints outside 10-550 mV are projected to the display boundary only. '
        'Blue and red lines show literature and combined nominal Pareto fronts; rising points and bounds are excluded. '
        'Audited same-recipe measurements are pooled once under ordinary catalyst names. No new HD repeats are used.\n',encoding='utf-8')
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    main()
