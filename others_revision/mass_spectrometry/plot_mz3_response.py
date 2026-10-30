"""Re-layout saved calibration results without refitting or changing data."""
from pathlib import Path
import argparse
from hashlib import sha256
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
from lxml import etree as E
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'inputs/mz3_response'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs/mz3_response')
OUT = parser.parse_args().output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parents[1] / '0-Figures'))
from figure_typography import normalize_tree, inkscape_executable

files = ['summary.json', 'calibration_points.csv', 'matched_response_points.csv', 'response_curves.csv']
hashes = {name: sha256((SOURCE / name).read_bytes()).hexdigest() for name in files}
summary = json.loads((SOURCE / 'summary.json').read_text(encoding='utf-8'))
calibration = pd.read_csv(SOURCE / 'calibration_points.csv')
points = pd.read_csv(SOURCE / 'matched_response_points.csv')
curves = pd.read_csv(SOURCE / 'response_curves.csv')
fit = summary['calibration_reproduction']['HD']
control_fit = next(f for f in summary['control_descriptive_fits'] if f['degree'] == 1)
assert len(calibration[calibration.species == 'HD']) == 21 and len(points) == 40

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11, 'font.weight': 'bold',
    'axes.labelweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
})
BLUE, ORANGE = '#167da6', '#bd5728'

def export(fig, name):
    svg = OUT / (name + '.svg')
    fig.savefig(svg, facecolor='white')
    plt.close(fig)
    tree = E.parse(str(svg))
    typography = normalize_tree(tree)
    tree.write(str(svg), encoding='utf-8', xml_declaration=True)
    for extension in ['png', 'pdf']:
        args = [str(inkscape_executable()), str(svg),
                '--export-area-page', '--export-background=white', '--export-background-opacity=1',
                '--export-filename=' + str(svg.with_suffix('.' + extension))]
        args.append('--export-width=2800' if extension == 'png' else '--export-text-to-path')
        subprocess.run(args, check=True, capture_output=True)
    return {'file': svg.name, 'typography_fixes': typography}

fig, (standard, control) = plt.subplots(1, 2, figsize=(10.0, 3.7))
fig.subplots_adjust(left=.085, right=.985, bottom=.18, top=.875, wspace=.29)
subset = calibration[calibration.species == 'HD']
xmin, xmax = fit['measured_concentration_range_volpct']
xline = np.linspace(0, xmax, 301)
yline = (fit['slope'] * xline + fit['intercept']) / 1e-10
standard.plot(xline, yline, color='#d62728', lw=1.5, ls=':', label='Fit extension')
standard.plot(xline[xline >= xmin], yline[xline >= xmin], color='#d62728', lw=1.7,
              label='Full linear fit')
for run, marker in [(1, 'o'), (2, 's'), (3, '^')]:
    p = subset[subset.run == run]
    standard.scatter(p.concentration_volpct, p.baseline_corrected_signal_mbar / 1e-10,
                     s=27, marker=marker, color=BLUE, edgecolors='white', linewidths=.4,
                     label=f'Run {run}', zorder=3)
standard.scatter([0], [0], marker='x', color='black', s=38, linewidths=1.2,
                 label='Origin: display only', zorder=4)
standard.set(xlabel='HD concentration (vol%)',
             ylabel=r'Baseline-corrected signal ($10^{-10}$ mbar)',
             xlim=(-.03 * xmax, 1.05 * xmax), ylim=(-.035 * yline.max(), 1.12 * yline.max()))
standard.set_xticks([0, .004, .008, .012])
standard.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
standard.legend(loc='upper left', fontsize=8.5, frameon=False, ncol=2,
                columnspacing=.8, handlelength=1.6)
standard.text(.97, .07, r'$I_3=6.16\times10^{-8}c_{\mathrm{HD}}+9.51\times10^{-12}$'
              + '\n' + rf'$R^2={fit["R2"]:.3f}$; n = 21', transform=standard.transAxes,
              ha='right', va='bottom', fontsize=10.5)

control.plot(curves.h2_concentration_volpct, curves.mz3_h2_linear_fit_net_mbar / 1e-11,
             color=ORANGE, lw=1.7, ls='--', label='Linear fit')
control.scatter(points.h2_concentration_volpct, points.mz3_h2_net_mbar / 1e-11,
                color=ORANGE, edgecolors='white', linewidths=.4, s=28,
                zorder=3, label='Measured controls')
control.set(xlabel=r'H$_2$ concentration (vol%)', ylabel=r'Net m/z = 3 signal ($10^{-11}$ mbar)',
            xlim=(.14, .38), ylim=(0, 2.2))
control.set_xticks([.16, .20, .24, .28, .32, .36])
control.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
control.legend(loc='upper left', frameon=False, fontsize=9.5)
control.text(.97, .07, r'$I_3=7.28\times10^{-11}c_{\mathrm{H_2}}-7.62\times10^{-12}$'
             + '\n' + rf'$R^2={control_fit["descriptive_R2"]:.3f}$; n = 40',
             transform=control.transAxes, ha='right', va='bottom', fontsize=10.5)
for ax, label in zip([standard, control], 'AB'):
    ax.grid(alpha=.13, linewidth=.5, axis='y')
    ax.text(-.18, 1.035, label, transform=ax.transAxes, fontsize=20, ha='left', va='bottom')
exports = [export(fig, 'hd_calibration_h2_control')]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
fig.subplots_adjust(left=.13, right=.98, bottom=.16, top=.82)
ax.plot(curves.hd_concentration_volpct, curves.mz3_hd_calibration_total_mbar,
        color=BLUE, lw=2, label='HD standard: full calibration')
visible = subset[subset.concentration_volpct.between(curves.hd_concentration_volpct.min(),
                                                    curves.hd_concentration_volpct.max())]
ax.scatter(visible.concentration_volpct, visible.baseline_corrected_signal_mbar,
           color=BLUE, edgecolors='white', linewidths=.4, s=30, zorder=4,
           label='HD standard: measured points')
ax.plot(curves.hd_concentration_volpct, curves.mz3_h2_linear_fit_net_mbar,
        color=ORANGE, lw=1.5, ls='--', label=r'H$_2$ control: linear fit')
ax.scatter(points.hd_concentration_volpct, points.mz3_h2_net_mbar,
           color=ORANGE, edgecolors='white', linewidths=.4, s=28, zorder=5,
           label=r'H$_2$ control: measured points')
ax.set(xlim=(curves.hd_concentration_volpct.min() - .00015,
             curves.hd_concentration_volpct.max() + .00015),
       xlabel='HD concentration (vol%)', ylabel='m/z = 3 signal (mbar)', yscale='log',
       ylim=(3e-12, 1.1e-9))
ax.set_xticks([.004, .005, .006, .007, .008])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ratio = summary['H2_to_HD_molecular_concentration_ratio']
top = ax.secondary_xaxis('top', functions=(lambda c: c * ratio, lambda c: c / ratio))
top.set_xlabel(r'Corresponding H$_2$ concentration (vol%)',
               fontproperties=ax.xaxis.label.get_fontproperties(), labelpad=7)
top.set_xticks([.16, .20, .24, .28, .32, .36])
top.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
top.tick_params(labelsize=11)
ax.yaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.grid(alpha=.15, linewidth=.5, axis='y')
ax.legend(loc='center left', frameon=False, fontsize=9.5, labelspacing=.35)
exports.append(export(fig, 'matched_mz3_responses'))

assert hashes == {name: sha256((SOURCE / name).read_bytes()).hexdigest() for name in files}
(OUT / 'plot_provenance.json').write_text(json.dumps({
    'sources': hashes, 'data_unchanged': True, 'refitted': False,
    'panel_mapping': {'A': 'HD calibration', 'B': 'H2 control', 'standalone': 'Matched m/z = 3 responses'},
    'exports': exports,
}, indent=2), encoding='utf-8')
print(json.dumps({'outputs': exports, 'data_unchanged': True}, indent=2))
