"""Check input integrity, NMR numerical references and MS figure consistency."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def verify_inputs():
    for folder in ['nmr', 'mass_spectrometry']:
        entries = json.loads((ROOT/folder/'input_manifest.json').read_text(encoding='utf-8'))
        for entry in entries:
            path = ROOT/folder/entry['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'], entry['path']

def verify():
    checks = []
    expected = json.loads((ROOT/'nmr/validation/accepted_numeric_reference.json').read_text(encoding='utf-8'))
    actual = json.loads((ROOT/'nmr/results/migration_results.json').read_text(encoding='utf-8'))
    numeric_count = 0
    max_error = 0.0

    def compare(a, b, loc):
        nonlocal numeric_count, max_error
        if isinstance(a, dict):
            for k in a:
                compare(a[k], b[k], loc+'/'+k)
        elif isinstance(a, list):
            assert len(a)==len(b), loc
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, loc+'/'+str(i))
        elif isinstance(a, (float, int)) and not isinstance(a, bool):
            assert np.isclose(a, b, rtol=1e-10, atol=1e-12), (loc, a, b)
            numeric_count += 1
            max_error = max(max_error, abs(a-b))
        else:
            assert a==b, (loc, a, b)

    for key in expected:
        compare(expected[key], actual[key], key)
    checks.append({'analysis':'NMR adopted results and robustness checks',
                   'numeric_values':numeric_count, 'max_absolute_difference':max_error})
    ms = ROOT/'mass_spectrometry/outputs'
    controls = pd.read_csv(ms/'hiden_pulse_windows.csv')
    matched = pd.read_csv(ms/'interference_sensitivity/matched_controls.csv')
    sensitivity = pd.read_csv(ms/'interference_sensitivity/sensitivity_by_control.csv')
    summary = pd.read_csv(ms/'interference_sensitivity/summary.csv')
    assert len(controls)==len(matched)==40
    assert list(controls.pulse)==list(matched.pulse)==list(range(1, 41))
    assert np.allclose(controls.h2_median_mbar, matched.h2, rtol=1e-10, atol=0)
    assert np.allclose(controls.net_m3_mbar, matched.net_h3, rtol=1e-10, atol=0)
    assert len(sensitivity)==400 and len(summary)==20
    assert set(sensitivity.true_alpha)=={2, 5, 10, 15, 20}
    assert set(sensitivity.scenario)=={'net_h3', 'raw_h3'}
    assert (sensitivity.apparent_alpha < sensitivity.true_alpha).all()
    assert np.allclose(sensitivity.downward_bias_percent,
        100*(sensitivity.true_alpha-sensitivity.apparent_alpha)/sensitivity.true_alpha)
    checks.append({'analysis':'hydrogen control and interference sensitivity', 'control_plateaus':40,
                   'sensitivity_rows':400, 'summary_rows':20})
    figures = [ROOT/'nmr/results/figures'/name for name in
               ['nmr_calibration', 'nmr_calibration_spectra_low', 'nmr_calibration_spectra_high',
                'stagewise_deuterium_enrichment', 'stage_product_nmr_spectra']]
    figures += [ms/'interference_sensitivity'/name
                for name in ['hiden_hydrogen_control', 'h3_separation_sensitivity']]
    for stem in figures:
        for ext in ['.png', '.pdf', '.svg']:
            assert stem.with_suffix(ext).is_file(), str(stem.with_suffix(ext))
    checks.append({'analysis':'Publication figure exports', 'figures':len(figures)})
    from mass_spectrometry.verify_workbook import verify as verify_ms_workbook
    checks.append(verify_ms_workbook())
    from mass_spectrometry.verify_mz3_response import verify as verify_mz3_response
    checks.append(verify_mz3_response())
    return checks

if __name__=='__main__':
    verify_inputs()
    print(json.dumps(verify(), indent=2))
