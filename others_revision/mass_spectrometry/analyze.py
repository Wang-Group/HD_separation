"""Extract the Hiden H2-only control plateaus used in the control and interference-sensitivity figures."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT/'raw/mass_calibration/hiden_data/file147.csv'
OUT = ROOT/'outputs'
OUT.mkdir(exist_ok=True)

h = pd.read_csv(RAW, skiprows=24)
t, h2, m3 = (h.iloc[:, i].to_numpy(float) for i in (1, 2, 3))
t = t/3600000
# Include all four-level dilution blocks before the later single-level pulses.
# These plateaus are not independent experimental replicates.
flag = (h2 > 1e-8) & (t >= 1.2) & (t <= 14)
starts = np.where(np.diff(flag.astype(int), prepend=0)==1)[0]
ends = np.where(np.diff(flag.astype(int), append=0)==-1)[0]
records = []
for a, b in zip(starts, ends):
    if b-a < 10:
        continue
    lo, hi = t[a]+.25*(t[b]-t[a]), t[a]+.75*(t[b]-t[a])
    core = (t >= lo) & (t <= hi)
    baseline = (t >= t[a]-.06) & (t <= t[a]-.02)
    records.append({'pulse':len(records)+1, 'start_h':t[a], 'end_h':t[b],
                    'core_start_h':lo, 'core_end_h':hi, 'n_core':int(core.sum()),
                    'h2_median_mbar':np.median(h2[core]),
                    'm3_median_mbar':np.median(m3[core]),
                    'm3_sd_mbar':np.std(m3[core], ddof=1),
                    'baseline_m3_mbar':np.median(m3[baseline]),
                    'net_m3_mbar':np.median(m3[core])-np.median(m3[baseline])})
controls = pd.DataFrame(records)
assert len(controls)==40
controls.to_csv(OUT/'hiden_pulse_windows.csv', index=False)
print(f'Extracted {len(controls)} Hiden control plateaus.')
