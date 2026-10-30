"""Plot observed work-function coordinates against the two saved eta10 targets."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

import plot_style  # noqa: F401
from workflow.data import load_data


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures/eta10_work_function_projection'
AUDIT = ROOT / 'results/eta10_target_audit.csv'
COORDS = ['work_function_element1', 'work_function_element2']
TARGETS = ['original_only_target_mV', 'averaged_target_mV']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data, _ = load_data()
    targets = pd.read_csv(AUDIT)
    targets = targets.loc[targets['included']].set_index('sample').sort_index()
    assert len(targets) == 298 and targets.index.is_unique
    plotted = targets.join(data[['element1', 'element2', 'actual_frac_element1'] + COORDS],
                           validate='one_to_one')
    fraction = plotted['actual_frac_element1']
    plotted['icp_weighted_work_function'] = (fraction * plotted[COORDS[0]]
                                           + (1 - fraction) * plotted[COORDS[1]])
    plotted['plotted'] = plotted[COORDS].notna().all(axis=1)
    assert plotted[TARGETS].notna().all().all()
    for source, capped in [('original_eta10_mV', TARGETS[0]), ('averaged_eta10_mV', TARGETS[1])]:
        np.testing.assert_allclose(plotted[source].clip(10, 550), plotted[capped])
    observed = plotted.loc[plotted['plotted']]
    assert len(observed) == 281
    assert observed['icp_weighted_work_function'].notna().all()
    assert observed['actual_frac_element1'].between(0, 1).all()
    OUT.mkdir(parents=True, exist_ok=True)
    plotted.to_csv(OUT / 'sample_coordinates_and_targets.csv', index=True)
    plotted.loc[~plotted['plotted']].to_csv(OUT / 'missing_work_function.csv', index=True)

    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold',
                         'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
    fig = plt.figure(figsize=(10.4, 8.8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045],
                           left=0.09, right=0.90, bottom=0.09, top=0.94,
                           wspace=0.46, hspace=0.56)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    norm = Normalize(10, 550)
    titles = ['A  Original-only', 'B  Averaged']
    # Shared ordering preserves exact observed coordinates; coincident points
    # are not jittered or converted into an interpolated prediction surface.
    for ax, target, title in zip(axes, TARGETS, titles):
        points = ax.scatter(observed[COORDS[0]], observed[COORDS[1]],
                            c=observed[target], cmap='coolwarm', norm=norm,
                            s=43, linewidths=0.5, edgecolors='#333333', zorder=3)
        ax.set(xlim=(2.4, 5.6), ylim=(2.4, 5.6),
               xticks=np.arange(2.5, 5.6, 0.5), yticks=np.arange(2.5, 5.6, 0.5))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Work Function of Metal 1 (eV)', labelpad=9)
        ax.set_ylabel('Work Function of Metal 2 (eV)', labelpad=8)
        ax.set_title(title, pad=12, fontsize=14)
        ax.tick_params(width=1, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
    for i, (target, title) in enumerate(zip(TARGETS, ['C  Original-only', 'D  Averaged'])):
        ax = fig.add_subplot(grid[1, i])
        ax.scatter(observed['icp_weighted_work_function'], observed[target],
                   c=observed[target], cmap='coolwarm', norm=norm,
                   s=32, linewidths=0.5, edgecolors='#333333', zorder=3)
        ax.set(xlim=(2.4, 5.6), ylim=(0, 570), xticks=np.arange(2.5, 5.6, 0.5),
               yticks=[10, 100, 200, 300, 400, 500, 550])
        ax.set_xlabel('ICP-Weighted Work Function (eV)', labelpad=9)
        ax.set_ylabel(r'$\mathbf{\eta_{10}}$ target (mV)', labelpad=8)
        ax.set_title(title, pad=12, fontsize=14)
        ax.tick_params(width=1, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
    colorbar = fig.colorbar(points, cax=fig.add_subplot(grid[:, 2]),
                           ticks=[10, 100, 200, 300, 400, 500, 550])
    colorbar.set_label(r'$\mathbf{\eta_{10}}$ target (mV)', labelpad=12)
    stem = OUT / 'eta10_work_function_projection'
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=300, facecolor='white')
    plt.close(fig)

    inputs = [AUDIT, ROOT / 'inputs/reference/elemental_properties.csv',
              ROOT / 'inputs/icp/ICP_raw_and_percentages_EN.xlsx',
              ROOT / 'workflow/data.py', ROOT / 'plot_style.py', Path(__file__)]
    manifest = {
        'command': 'python plot_eta10_work_function_projection.py',
        'working_directory': 'ml_revision', 'model_fit': False,
        'cohort_n': len(plotted), 'plotted_n': len(observed),
        'missing_coordinates_n': int((~plotted['plotted']).sum()),
        'unique_observed_coordinates': len(observed[COORDS].drop_duplicates()),
        'missing_coordinate_samples': plotted.index[~plotted['plotted']].tolist(),
        'coordinate_policy': 'Observed element-level work functions; no imputation or jitter. '
                             'Points with identical coordinates overlap; both panels share sample order.',
        'target_columns': TARGETS, 'shared_color_limits_mV': [10, 550],
        'weighted_work_function': 'f_M1 * WF_M1 + (1 - f_M1) * WF_M2; f_M1 is the ICP molar fraction',
        'target_policy': 'Saved crossing-only targets; BO-reference measurements excluded from averages.',
        'input_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs},
        'output_sha256': {p.name: sha256(p) for p in sorted(OUT.iterdir())
                          if p.suffix in {'.png', '.svg', '.pdf', '.csv'}},
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ['cohort_n', 'plotted_n',
                                             'missing_coordinates_n', 'model_fit']}, indent=2))


if __name__ == '__main__':
    main()
