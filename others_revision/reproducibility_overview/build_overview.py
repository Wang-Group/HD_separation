"""Joint reproducibility overview from current LSV results and retained H/D data."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / '0-Figures/Figure1/revision_figure'))
from lsv_plot_data import load_data, input_hashes

HD_SOURCE = HERE.parent / 'HD_repeat_analysis/tables/sample_statistics.csv'
HD_TO_LSV = {}
STYLES = {
    'both': dict(color='#208b76', marker='s', label='Both SDs'),
    'potential_only': dict(color='#4175b0', marker='o', label='Potential SD only'),
    'HD_only': dict(color='#d67827', marker='^', label='H/D SD only'),
}


def assemble():
    _, lsv = load_data()
    hd = pd.read_csv(HD_SOURCE)
    hd['HD_source_sample'] = hd['sample']
    hd['sample'] = hd['sample'].replace(HD_TO_LSV)
    joined = lsv.merge(hd, on='sample', how='outer', validate='one_to_one', indicator=True)
    assert joined['_merge'].eq('both').all()
    selected = joined.loc[joined.potential_SD_mV.notna() | joined.SD.notna()].copy()
    selected = selected.rename(columns={'mean_S_HD': 'HD_mean', 'SD': 'HD_SD',
        'n_measurements': 'n_HD', 'potential_mean_mV': 'plotted_potential_mean_mV',
        'potential_SD_mV': 'plotted_potential_SD_mV', 'n_potential': 'n_plotted_potential',
        'categories': 'HD_source_categories'})
    has_x, has_y = selected.plotted_potential_SD_mV.notna(), selected.HD_SD.notna()
    selected['SD_category'] = np.select([has_x & has_y, has_x], ['both', 'potential_only'], default='HD_only')
    selected['eta10_status'] = np.select(
        [selected.potential_metric.eq('estimate'), selected.potential_metric.eq('scan_limit'),
         selected.potential_metric.eq('no_accepted_result'), selected.n_eta10.eq(1)],
        ['Estimate only; no measured eta10', 'Scan limit; no measured eta10',
         'No accepted result', 'One eta10; SD unavailable'], default='eta10 mean and SD')
    selected['HD_status'] = np.where(has_y, 'H/D mean and SD', 'One H/D value; SD unavailable')
    selected['plot_location'] = np.where(selected.plotted_potential_mean_mV.notna(), 'main', 'missing_x')
    # Missing coordinates must not silently receive zero or a boundary estimate.
    assert selected.plot_location.eq('main').all(), 'Add a marginal panel for newly missing coordinates'
    selected = selected.drop(columns='_merge').sort_values('sample')
    rows = selected.astype(object).where(selected.notna(), None).to_dict('records')
    return rows, Counter(selected.SD_category)


def plot(rows, counts):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})
    fig, ax = plt.subplots(figsize=(11, 7.2))
    fig.subplots_adjust(left=.10, right=.98, bottom=.18, top=.87)
    xlow = min(r['plotted_potential_mean_mV'] - (r['plotted_potential_SD_mV'] or 0) for r in rows)
    xhigh = max(r['plotted_potential_mean_mV'] + (r['plotted_potential_SD_mV'] or 0) for r in rows)
    ylow = min(r['HD_mean'] - (r['HD_SD'] or 0) for r in rows)
    yhigh = max(r['HD_mean'] + (r['HD_SD'] or 0) for r in rows)
    for category, style in STYLES.items():
        selected = [r for r in rows if r['SD_category'] == category]
        for r in selected:
            ax.errorbar(r['plotted_potential_mean_mV'], r['HD_mean'],
                xerr=r['plotted_potential_SD_mV'], yerr=r['HD_SD'], fmt='none',
                ecolor=style['color'], alpha=.34, elinewidth=.8, capsize=1.8, zorder=1)
            estimate = r['potential_metric'] == 'estimate'
            ax.scatter(r['plotted_potential_mean_mV'], r['HD_mean'],
                s=58 if estimate else 26 if category == 'potential_only' else 37,
                marker='D' if estimate else style['marker'],
                facecolor='white' if estimate else style['color'],
                edgecolor=style['color'] if estimate else 'white',
                linewidth=1.3 if estimate else .35, alpha=.9, zorder=5 if estimate else 3)
    xlim = (np.floor(xlow / 25) * 25, np.ceil(xhigh / 25) * 25)
    ylim = (np.floor(ylow - .2), np.ceil(yhigh + .2))
    ax.set(xlim=xlim, ylim=ylim, xlabel='Mean overpotential (mV)', ylabel='Mean H/D separation coefficient')
    ax.grid(color='#e6e6e6', lw=.6, alpha=.6)
    ax.set_axisbelow(True)
    offsets = {'Co-V11': (32, -35), 'Eu-Pd': (55, 12), 'W-Pd': (45, 25)}
    for r in rows:
        if r['potential_metric'] != 'eta10':
            label = r['sample'] + '\nestimate, not $\\eta_{10}$'
            ax.annotate(label, (r['plotted_potential_mean_mV'], r['HD_mean']),
                xytext=offsets.get(r['sample'], (15, 20)), textcoords='offset points', fontsize=10,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2),
                arrowprops=dict(arrowstyle='-', color='#555555', lw=.8), zorder=6)
    handles = [Line2D([], [], marker=STYLES[k]['marker'], ls='none', color=STYLES[k]['color'],
        markersize=7, label=f"{STYLES[k]['label']} ({counts[k]})") for k in ['both','potential_only','HD_only']]
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.53,.99),
               frameon=False, fontsize=11.5, columnspacing=2.4, handletextpad=.5)
    fig.text(.10,.075, r'Open diamonds: estimate means, not measured $\eta_{10}$. Bars: sample SD, not SEM.', fontsize=10.5)
    fig.text(.10,.035, 'Missing error bars mean fewer than two values, not zero variability.', fontsize=10.5)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(HERE / f'reproducibility_overview.{extension}', dpi=260, facecolor='white')
    plt.close(fig)
    return dict(x_limits=list(xlim), y_limits=list(ylim))


def main():
    before = input_hashes()
    before['HD_statistics'] = hashlib.sha256(HD_SOURCE.read_bytes()).hexdigest()
    rows, counts = assemble()
    pd.DataFrame(rows).to_csv(HERE / 'overview_points.csv', index=False)
    pd.DataFrame(rows).iloc[0:0].to_csv(HERE / 'missing_or_unassigned_coordinates.csv', index=False)
    bounds = plot(rows, counts)
    summary = dict(total_catalysts=len(rows), SD_categories=dict(counts),
        numeric_scatter_points=len(rows), marginal_points=[], input_sha256=before,
        HD_to_LSV_aliases=HD_TO_LSV, display_limits=bounds,
        measured_eta10_SD_catalysts=sum(r['eta10_SD_mV'] is not None for r in rows),
        estimate_SD_catalysts=sum(r['potential_metric']=='estimate' and r['plotted_potential_SD_mV'] is not None for r in rows),
        estimate_samples=[r['sample'] for r in rows if r['potential_metric']=='estimate'],
        HD_SD_catalysts=sum(r['HD_SD'] is not None for r in rows),
        policy='Union of catalysts with an eta10 SD, estimate-potential SD, or H/D SD. '
        'Crossing means are preferred; estimates are only used without accepted crossings and are labelled separately. '
        'Audited same-recipe measurements pooled once using the same LSV aggregation as Figure 1B. '
        'H/D repeat means/SDs use adopted individual coefficients; Figure 1B instead uses initial-submission H/D values. '
        'Uncapped means and sample SDs. No Pareto analysis or ML.')
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    assert before == {**input_hashes(), 'HD_statistics': hashlib.sha256(HD_SOURCE.read_bytes()).hexdigest()}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
