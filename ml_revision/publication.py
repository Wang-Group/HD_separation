"""Rebuild publication figures and tables from saved nested-validation records."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image
from plot_helpers import ROOT, OUT, take, attribution, display, compact_label, main_shap, multi_bars, save_plot
from workflow.data import FEATURES
from model_core import digest, save
from figure_typography import normalize_exports

TABLES = ROOT / 'tables'
PROTOCOLS = [('stratified_random', 'Random'), ('unseen_pair', 'Unseen pair')]
ARMS = [('without_bo', 'Without BO'), ('with_bo', 'With BO')]

def metric(row, key):
    return f'{row[key+"_mean"]:.2f} +/- {row[key+"_sd"]:.2f}'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.labelweight': 'bold',
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})
    phi_path = ROOT / 'results/shap.csv.gz'
    if not phi_path.exists():
        phi_path = ROOT / 'results/shap.csv'
    phi = pd.read_csv(phi_path)
    pred = pd.read_csv(ROOT/'results/predictions.csv')
    metrics = pd.read_csv(ROOT/'results/repeat_metrics.csv')
    summary = pd.read_csv(ROOT/'results/summary.csv')
    settings = pd.DataFrame(json.loads((ROOT/'results/settings.json').read_text()))
    rows = []
    for (wf, protocol, arm, strategy), g in phi.groupby(['workflow','protocol','arm','strategy']):
        fitted = take(settings, wf, protocol, arm, strategy)
        for rank, (feature, r) in enumerate(attribution(g).iterrows(), 1):
            rows.append(dict(workflow=wf, protocol=protocol, arm=arm, strategy=strategy, feature=feature,
                name=display(feature), rank=rank, mean_absolute_shap=r['mean'], repeat_sd=r.sd,
                selected_folds=sum(feature in x for x in fitted.features), total_folds=len(fitted)))
    features = pd.DataFrame(rows)
    features.to_csv(TABLES/'feature_importance_and_frequency.csv', index=False)
    summary.to_csv(TABLES/'performance_full_precision.csv', index=False)
    settings.groupby(['workflow','protocol','arm','strategy']).n_features.agg(['min','max','mean']).to_csv(TABLES/'selected_input_counts.csv')
    figures = []
    with PdfPages(OUT/'predictive_figures.pdf') as pdf:
        main_shap(phi,pred,metrics,settings,'hd_icp','stratified_random','without_bo','icp','hd_shap_random','',pdf,figures)
        main_shap(phi,pred,metrics,settings,'hd_icp','unseen_pair','without_bo','icp','hd_shap_unseen','',pdf,figures)
        multi_bars(phi, [(f'{letter}  {plabel}, {alabel.lower()}', 'hd_icp', arm, protocol)
            for letter,(protocol,plabel,arm,alabel) in zip('ABCD',
                [(p,pl,a,al) for p,pl in PROTOCOLS for a,al in ARMS])],
            'hd_validation_shap','',pdf,figures)
        multi_bars(phi,[('A  Unshifted (30 inputs)','hd_icp','without_bo','stratified_random'),
            ('B  Unshifted + measurements (33)','ec_unshifted_measured','without_bo','stratified_random'),
            ('C  Potential-shifted (30)','ec_shifted','without_bo','stratified_random'),
            ('D  Shifted + measurements (33)','ec_shifted_measured','without_bo','stratified_random')],
            'electrochemical_shap','',pdf,figures)
        for protocol, label in PROTOCOLS:
            for arm in ['original_only','averaged']:
                main_shap(phi,pred,metrics,settings,'eta10',protocol,arm,'icp',f'eta10_{protocol}_{arm}','',pdf,figures)
        fig, axes = plt.subplots(1,4,figsize=(10.5,9.0),sharey=True)
        fig.subplots_adjust(left=.29,right=.99,bottom=.07,top=.94,wspace=.12)
        for ax,(protocol,plabel,arm,alabel) in zip(axes,[(p,pl,a,al) for p,pl in PROTOCOLS for a,al in ARMS]):
            f=take(features,'hd_icp',protocol,arm).set_index('feature').reindex(FEATURES)
            y=np.arange(len(f))[::-1]
            ax.barh(y,f.selected_folds,color='#197ca5',height=.72)
            ax.set(xlim=(0,15.7),xticks=[0,5,10,15],xlabel='Selected fits / 15',title=plabel+'\n'+alabel)
            ax.set_yticks(y,[compact_label(x) for x in FEATURES],fontsize=12.5)
            ax.tick_params(axis='y',length=0)
            ax.tick_params(axis='x',labelsize=12)
            ax.set_xlabel('Fits / 15',fontsize=13)
            ax.set_title(plabel+'\n'+alabel,fontsize=13,fontweight='bold')
            ax.spines[['top','right']].set_visible(False)
        save_plot(fig,'hd_selection_frequency','',pdf,figures)
    typography=normalize_exports(OUT,apply=True)
    save(ROOT/'checks/figure_typography.json',dict(corrections=typography,remaining=normalize_exports(OUT)))
    from pypdf import PdfWriter
    combined=PdfWriter()
    for entry in figures:combined.append(OUT/(entry['file']+'.pdf'))
    with (OUT/'predictive_figures.pdf').open('wb') as f:combined.write(f)
    # One SI image per target comparison, using the corrected component figures.
    for protocol,_ in PROTOCOLS:
        files=[OUT/f'eta10_{protocol}_{arm}.png' for arm in ['original_only','averaged']]
        images=[Image.open(f).convert('RGB') for f in files]
        joined=Image.new('RGB',(images[0].width,sum(x.height for x in images)+50),'white')
        y=0
        from PIL import ImageDraw, ImageFont
        draw=ImageDraw.Draw(joined)
        from matplotlib import font_manager
        font=ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family='Arial',weight='bold'),fallback_to_default=False),38)
        for image,label in zip(images,['A  Initial LSV target','B  Averaged target']):
            joined.paste(image,(0,y+25));draw.text((28,y+4),label,font=font,fill='black');y+=image.height+25
        joined.save(OUT/f'eta10_{protocol}_comparison.png')
    # Word tables are built from these rows; no figures or document numbers are file identifiers.
    tables={}
    def performance_rows(wfs, arms, strategies, representation=False):
        rows=[]
        for p,pl in PROTOCOLS:
            for wf,wl in wfs:
                for a,al in arms:
                    for strategy,sl in strategies:
                        q=take(summary,wf,p,a,strategy)
                        if q.empty: continue
                        r=q.iloc[0]
                        rows.append([pl,wl if representation else al,sl,metric(r,'r2'),metric(r,'rmse')])
        return rows
    strategies=[('correlation90','Selected'),('all_candidates','All inputs')]
    tables['hd_validation']=[['Protocol','Training arm','Inputs','R2','RMSE']]+performance_rows([('hd_icp','ICP')],ARMS,strategies)
    tables['representation']=[['Protocol','Representation','Inputs','R2','RMSE']]+performance_rows(
        [('hd_precursor','Precursor (30)'),('hd_icp','ICP (30)'),('hd_fraction','ICP + M1 fraction (31)')],
        [('with_bo','With BO')],strategies,True)
    tables['electrochemical']=[['Configuration','Inputs','R2','RMSE']]
    for wf,wl in [('hd_icp','Unshifted (30)'),('ec_unshifted_measured','Unshifted + measured (33)'),('ec_shifted','Shifted (30)'),('ec_shifted_measured','Shifted + measured (33)')]:
        for strategy,sl in strategies:
            r=take(summary,wf,strategy=strategy).iloc[0]
            tables['electrochemical'].append([wl,sl,metric(r,'r2'),metric(r,'rmse')])
    tables['eta10']=[['Protocol','Target','Inputs','R2','RMSE (mV)']]+performance_rows([('eta10','eta10')],
        [('original_only','Initial LSV'),('averaged','Averaged')],strategies)
    header=['Descriptor','Random no BO','Random with BO','Unseen no BO','Unseen with BO']
    tables['selection_frequency']=[header];tables['shap']=[header]
    for feature in FEATURES:
        freq=[display(feature,True)];values=[display(feature,True)]
        for p,_ in PROTOCOLS:
            for a,_ in ARMS:
                r=take(features,'hd_icp',p,a).set_index('feature').loc[feature]
                freq.append(f'{int(r.selected_folds)}/15')
                values.append(f'{r.mean_absolute_shap:.2f} +/- {r.repeat_sd:.2f}')
        tables['selection_frequency'].append(freq);tables['shap'].append(values)
    save(TABLES/'si_table_rows.json',tables)
    save(ROOT/'FIGURE_INDEX.json',dict(prediction_figures=figures,
        sources=['results/shap.csv.gz','results/predictions.csv','results/repeat_metrics.csv'],
        reproduction='python publication.py', font='Arial', fit_performed=False,
        hashes={p.relative_to(ROOT).as_posix():digest(p) for p in OUT.iterdir() if p.suffix in ['.png','.pdf','.svg']}))
    print('Publication figures and tables exported without model fitting.')

if __name__ == '__main__': main()
