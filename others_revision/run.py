"""Reproduce the packaged NMR and MS response figures without execution archives."""
from pathlib import Path
import argparse
import subprocess
import sys

from verify_results import verify, verify_inputs

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify_inputs()
    if not args.verify_only:
        scripts = ['nmr/run_analysis.py', 'nmr/export_tables.py', 'nmr/plot_response.py',
                   'nmr/plot_calibration_spectra.py',
                   'mass_spectrometry/analyze.py',
                   'mass_spectrometry/interference_sensitivity.py',
                   'mass_spectrometry/plot_response.py',
                   'mass_spectrometry/plot_mz3_response.py']
        for script in scripts:
            result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/script)],
                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
            print(f'{script}: {result.returncode}', flush=True)
            if result.returncode:
                raise RuntimeError(f'{script}\n{result.stdout}\n{result.stderr}')
    checks = verify()
    verify_inputs()
    print(f'All {len(checks)} result checks passed.')


if __name__ == '__main__':
    main()
