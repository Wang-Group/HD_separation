"""Publication plotting helpers for saved held-out predictions and SHAP."""
import html
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import shap
from descriptors import COVARIATES
from model_core import save,digest
from workflow.data import ROOT, BASES, RAW, load_data

OUT=ROOT/'figures/prediction'
RESULTS=ROOT/'results'
NAMES={'work_function':'Work Function','FE_ads_avg':'H Adsorption Free Energy',
       'Frequency_H':'H Vibrational Frequency','pzc':'Potential of Zero Charge',
       'E_std':'Standard Potential','electronegativity':'Electronegativity',
       'delta_EOH':'OH Adsorption Energy','MM_bond':'M-M Bond Energy',
       'MH_bond_kJ':'M-H Bond Energy','Frequency_OH':'O-H Vibrational Frequency'}
WORKFLOWS={'hd_icp':'H/D: ICP weighting','hd_precursor':'H/D: precursor weighting',
           'hd_fraction':'H/D: ICP + explicit M1 fraction','ec_unshifted_measured':'Unshifted + measurements',
           'ec_shifted':'Potential-shifted','ec_shifted_measured':'Shifted + measurements','eta10':'eta10'}


def display(f, short=False):
    special={'actual_frac_element1':'ICP M1 Molar Fraction','I_raw':'Current near -1 V',
             'Rs_from_EIS':'EIS Resistance','E_IR_corrected_final':'Corrected Operating Potential'}
    if f in special:return special[f]
    for prefix,label in [('icp_weighted_','ICP-weighted '),('precursor_weighted_','Precursor-weighted ')]:
        if f.startswith(prefix):return label+NAMES[f.removeprefix(prefix)]
    base=f.removesuffix('_element1').removesuffix('_element2')
    return NAMES[base]+(' | M' if short else ' of Metal ')+('1' if f.endswith('element1') else '2')


def compact_label(feature):
    label=display(feature,True)
    for prefix,suffix in [('ICP-weighted ',' | ICP'),('Precursor-weighted ',' | precursor')]:
        if label.startswith(prefix):label=label.removeprefix(prefix)+suffix
    for long,short in [('H Adsorption Free Energy','H ads. free energy'),
                       ('H Vibrational Frequency','H frequency'),
                       ('O-H Vibrational Frequency','O-H frequency'),
                       ('Potential of Zero Charge','PZC'),
                       ('Corrected Operating Potential','Corrected potential')]:
        label=label.replace(long,short)
    return label


def take(frame,workflow,protocol='stratified_random',arm='without_bo',strategy='correlation90'):
    return frame.loc[frame.workflow.eq(workflow)&frame.protocol.eq(protocol)&frame.arm.eq(arm)&frame.strategy.eq(strategy)].copy()


def observed(scheme):
    data,_=load_data()
    data=data.loc[data.cohort.eq('initial')]
    out=data[RAW].copy()
    f=data.nominal_frac_element1 if scheme=='precursor' else data.actual_frac_element1
    prefix='precursor_weighted_' if scheme=='precursor' else 'icp_weighted_'
    for b in BASES:
        out[prefix+b]=(f*data[b+'_element1']).where(f.ne(0),0)+((1-f)*data[b+'_element2']).where(f.ne(1),0)
    cov=pd.read_csv(ROOT/'inputs/paired/sample_covariates.csv').set_index('sample')
    if scheme in ['shifted','shifted_measured']:
        offset=data['sample'].map(cov.E_IR_corrected_final)+.098
        for b,sign in [('work_function',1),('pzc',-1),('E_std',-1)]:
            for c in [b+'_element1',b+'_element2','icp_weighted_'+b]:out[c]+=sign*offset
    if scheme.endswith('_measured'):
        for c in COVARIATES:out[c]=data['sample'].map(cov[c])
    if scheme=='fraction':out['actual_frac_element1']=f
    return out


def save_plot(fig,name,title,pdf,figures):
    from matplotlib.text import Text
    for text in fig.findobj(match=Text):
        if 'eta10' in text.get_text():
            text.set_text(text.get_text().replace('eta10',r'$\eta_{10}$'))
    fig.savefig(OUT/(name+'.png'),dpi=170,facecolor='white')
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    pdf.savefig(fig,facecolor='white')
    figures.append(dict(file=name,title=title))
    plt.close(fig)


def attribution(shap_frame):
    per=shap_frame.assign(absolute=shap_frame.value.abs()).groupby(['repeat','feature']).absolute.mean().reset_index()
    return per.groupby('feature').absolute.agg(mean='mean',sd='std').sort_values('mean',ascending=False)


def main_shap(phi,pred,metrics,settings,workflow,protocol,arm,scheme,name,title,pdf,figures):
    f=take(phi,workflow,protocol,arm)
    p=take(pred,workflow,protocol,arm)
    m=take(metrics,workflow,protocol,arm)
    counts=take(settings,workflow,protocol,arm).n_features
    stats=attribution(f)
    stats.to_csv(OUT/(name+'_all_features.csv'))
    top=stats.head(10)
    features=top.index.tolist()
    signed=f.groupby(['sample','feature']).value.mean().unstack().reindex(columns=features)
    colors=observed(scheme).reindex(signed.index)[features]
    signed.to_csv(OUT/(name+'_signed_sample_means.csv'))
    colors.to_csv(OUT/(name+'_observed_colors.csv'))
    fig,(left,right)=plt.subplots(1,2,figsize=(11.5,5.8),gridspec_kw={'width_ratios':[1.12,1]})
    fig.subplots_adjust(left=.26,right=.965,bottom=.14,top=.97,wspace=.10)
    ypos=np.arange(len(top))[::-1]
    left.barh(ypos,top['mean'],xerr=top.sd,color='#0095f6',height=.68,
              error_kw={'ecolor':'#40505b','elinewidth':.8,'capsize':2})
    left.set_yticks(ypos,[compact_label(x) for x in features],fontsize=14.5,fontweight='bold')
    upper=top['mean']+top.sd
    left.set_xlim(0,float(max(upper.max()*1.04,upper.iloc[5:].max()/.43)))
    left.set_ylim(-1,10)
    left.set_xlabel('Mean absolute SHAP'+(' (mV)' if workflow=='eta10' else ''),fontsize=15)
    left.tick_params(axis='x',labelsize=14)
    left.xaxis.set_major_locator(MaxNLocator(4))
    left.xaxis.set_major_formatter(FormatStrFormatter('%.0f' if workflow=='eta10' else '%.2f'))
    left.tick_params(axis='y',length=0,pad=7)
    left.spines[['top','right']].set_visible(False)
    left.spines['left'].set_color('#cccccc')
    np.random.seed(42)
    explanation=shap.Explanation(values=signed.to_numpy(),data=colors.to_numpy(),feature_names=[display(x) for x in features])
    shap.plots.beeswarm(explanation,max_display=10,order=np.arange(10),ax=right,plot_size=None,
        show=False,color=shap.plots.colors.red_blue,s=9,alpha=.85,color_bar=True,
        color_bar_label='Observed Feature Value',group_remaining_features=False)
    right.set_yticklabels([])
    right.set_xlabel('Held-out SHAP'+(' (mV)' if workflow=='eta10' else ''),fontsize=15)
    right.tick_params(axis='x',labelsize=14)
    for colorbar in fig.axes[2:]:
        colorbar.tick_params(labelsize=13)
        colorbar.yaxis.label.set_size(14)
    right.set_ylim(left.get_ylim())
    right.xaxis.set_major_locator(MaxNLocator(4))
    right.xaxis.set_major_formatter(FormatStrFormatter('%.0f' if workflow=='eta10' else '%.2f'))
    inset=left.inset_axes([.53,.12,.45,.35])
    inset.scatter(p.y_true,p.y_pred,s=6,color='#edae36',edgecolors='none',alpha=.22,rasterized=True)
    lo,hi=(10,550) if workflow=='eta10' else (min(p.y_true.min(),p.y_pred.min())-.2,max(p.y_true.max(),p.y_pred.max())+.2)
    inset.plot([lo,hi],[lo,hi],'--',color='red',lw=1)
    inset.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Measured'+(' (mV)' if workflow=='eta10' else ''),ylabel='Predicted')
    inset.set_title(f'$R^2$ = {m.r2.mean():.2f} $\\pm$ {m.r2.std():.2f}\nRMSE = {m.rmse.mean():.2f} $\\pm$ {m.rmse.std():.2f}',fontsize=12,pad=5)
    inset.tick_params(labelsize=11)
    inset.xaxis.label.set_size(12);inset.yaxis.label.set_size(12)
    inset.xaxis.set_major_locator(MaxNLocator(3));inset.yaxis.set_major_locator(MaxNLocator(3))
    inset.grid(alpha=.15,lw=.4)
    for spine in inset.spines.values():spine.set_color('#138e94')
    save_plot(fig,name,title,pdf,figures)


def multi_bars(phi,panels,name,title,pdf,figures):
    n=len(panels)
    fig,axes=plt.subplots(2 if n>2 else 1,2,figsize=(11.2,8 if n>2 else 4.6),squeeze=False)
    fig.subplots_adjust(left=.205,right=.985,bottom=.09,top=.94 if n>2 else .88,wspace=.77,hspace=.34)
    colors=['#137ca7','#ce6b1d','#069777','#bb6095']
    for ax,(caption,workflow,arm,protocol),color in zip(axes.flat,panels,colors):
        stat=attribution(take(phi,workflow,protocol,arm)).head(10 if name=='hd_validation_shap' else 8)
        y=np.arange(len(stat))[::-1]
        ax.barh(y,stat['mean'],xerr=stat.sd,height=.7,color=color,error_kw={'ecolor':'#333333','elinewidth':.8,'capsize':2})
        ax.set_yticks(y,[compact_label(f) for f in stat.index],fontsize=13.5)
        ax.set_title(caption,fontsize=14,pad=8,fontweight='bold',loc='right')
        ax.set_xlabel('Mean absolute SHAP',fontsize=14)
        ax.tick_params(axis='x',labelsize=13)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='y',length=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    save_plot(fig,name,title,pdf,figures)
