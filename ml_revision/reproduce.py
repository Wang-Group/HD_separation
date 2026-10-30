"""Reproduce the adopted analysis or verify saved records without fitting."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def run(args):
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--jobs',type=int,default=6)
    parser.add_argument('--atlas',action='store_true',help='Also regenerate all 15 descriptive property pairs.')
    args=parser.parse_args()
    if args.check_only:
        run(['validate.py','--check-only','--jobs',str(args.jobs)])
        run(['verify_publication.py'])
        return
    if not args.plots_only:
        run(['-m','unittest','test_refit','-v'])
        run(['audit_electrochemical.py'])
        run(['refit.py','--jobs',str(args.jobs)])
        run(['validate.py','--jobs',str(args.jobs)])
        run(['pack_shap.py'])
    for name in ['publication.py','descriptor_correlations.py','observed_correlations.py','plot_composition.py',
                 'plot_eta10_work_function_projection.py','plot_weighted_trio_letter.py']:
        run([name])
    if args.atlas:run(['plot_six_weighted_descriptor_pairs.py'])
    run(['figure_typography.py',str(ROOT/'figures'),'--apply'])
    run(['verify_publication.py'])

if __name__=='__main__':main()
