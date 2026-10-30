"""Export the adopted two-panel Hiden control view (Hiden hydrogen control)."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/interference_sensitivity'
data = pd.read_csv(ROOT/'raw/mass_calibration/hiden_data/file147.csv', skiprows=24)
t, h2, m3 = (data.iloc[:, i].to_numpy(float) for i in (1, 2, 3))
t = t/3600000
controls = pd.read_csv(ROOT/'outputs/hiden_pulse_windows.csv')
assert len(controls) == 40
window = (t >= 1.2) & (t <= 2.5)
plt.rcParams.update({'font.family':'Arial', 'font.size':11,
    'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42,
    'svg.fonttype':'none', 'mathtext.fontset':'custom', 'mathtext.rm':'Arial',
    'mathtext.it':'Arial:italic', 'mathtext.bf':'Arial:bold'})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
for ax, y, label, color in zip(axes, (h2, m3), ('2', '3'), ('#245d87', '#9b3659')):
    ax.plot(t[window], y[window], color=color, lw=1)
    ax.set(xlabel='Time (h)', ylabel=f'm/z = {label} (mbar)',
           title=('A' if label=='2' else 'B')+f'  H$_2$-only control: m/z = {label}')
    ax.grid(alpha=.15)
axes[1].axhline(np.median(controls.baseline_m3_mbar), color='gray', ls='--', lw=1,
               label='Median pre-injection background')
axes[1].legend(fontsize=9)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT/f'hiden_hydrogen_control.{ext}', dpi=240)
plt.close(fig)
