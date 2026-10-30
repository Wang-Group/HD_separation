"""Plot the verified H/D SD histogram and cumulative distribution in Arial."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    font_manager.findfont('Arial', fallback_to_default=False)
    with (HERE / 'tables/sample_statistics.csv').open(encoding='utf-8', newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if int(row['n_measurements']) >= 2]
    summary = json.loads((HERE / 'tables/summary.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    values = np.array([float(row['SD']) for row in rows])
    ordered = np.sort(values)
    width = .1
    bins = np.arange(0, np.ceil(values.max() / width) * width + width / 2, width)
    axes[0].hist(values, bins=bins, color='#26849c', edgecolor='white', linewidth=.6)
    axes[0].axvline(np.median(values), color='#c94b35', ls='--', lw=1.3,
                   label=f'Median: {np.median(values):.2f}')
    axes[0].axvline(values.mean(), color='#343434', ls=':', lw=1.3,
                   label=f'Mean: {values.mean():.2f}')
    axes[0].set_ylabel('Number of catalysts')
    axes[0].legend(frameon=False, loc='upper right', fontsize=11)
    cumulative = np.arange(1, len(values) + 1) / len(values) * 100
    axes[1].step(np.r_[0, ordered], np.r_[0, cumulative], where='post', color='#26849c', lw=1.8)
    # Round the observed 95% cutoff upward to a readable reporting threshold.
    cutoff95 = float(ordered[int(np.ceil(.95 * len(values))) - 1])
    reporting_cutoff = float(np.ceil(cutoff95 * 10) / 10)
    threshold_rows = []
    for threshold, color, ytext in [(1., '#555555', 50), (reporting_cutoff, '#c94b35', 74)]:
        count = int((values <= threshold).sum())
        percent = 100 * count / len(values)
        axes[1].plot([threshold, threshold], [0, percent], ls='--', color=color, lw=1)
        axes[1].plot(threshold, percent, 'o', color=color, ms=4)
        axes[1].text(1.6, ytext, f'{percent:.2f}% with\nSD \u2264 {threshold:.1f}',
                     fontsize=10.5, color=color, va='center')
        threshold_rows.append(dict(SD_threshold=threshold, catalysts=count, percent=percent))
    axes[1].set(ylabel='Cumulative proportion (%)', ylim=(0, 103))
    for ax, label in zip(axes, ['A', 'B']):
        ax.set(xlabel='Sample SD of H/D separation coefficient', xlim=(0, bins[-1]))
        ax.text(-.14, 1.05, label, transform=ax.transAxes, fontsize=15, fontweight='bold')
    fig.subplots_adjust(left=.08, right=.99, bottom=.20, top=.90, wspace=.28)
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'HD_repeatability_distribution.{extension}', dpi=240)
    plt.close(fig)
    with (HERE / 'tables/SD_distribution_thresholds.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    assert len(rows) == summary['n_repeat_groups']
    print(out / 'HD_repeatability_distribution.png')


if __name__ == '__main__':
    main()
