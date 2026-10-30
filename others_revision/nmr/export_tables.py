"""Export adopted NMR concentration and enrichment tables without changing reported precision."""
from pathlib import Path
import argparse
import csv
import json

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, default=HERE/'results')
args = parser.parse_args()
data = json.loads((args.results/'migration_results.json').read_text(encoding='utf-8'))
config = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
out = args.results/'tables'
out.mkdir(exist_ok=True)
products = sorted([r for r in data['concentrations'] if r['role']=='product'], key=lambda r:r['stage'])
r5 = [{'Stage':str(r['stage']), 'Group':r['group'],
    'Product D (atom%)':f"{r['saved_1r_D_percent']:.4f}",
    'Calibration 95% interval':f"{r['saved_1r_calibration_95_low']:.4f} to {r['saved_1r_calibration_95_high']:.4f}",
    'Model plus reading 95% interval':f"{r['saved_1r_model_reading_95_low']:.4f} to {r['saved_1r_model_reading_95_high']:.4f}"}
    for r in products]
r6 = []
for r, vi, vf in zip(data['enrichment'], config['initial_volumes_ml'], config['final_volumes_ml']):
    r6.append({'Stage':str(r['stage']), 'Initial to final volume (mL)':f'{vi:g} -> {vf:g}',
        'Feed D (atom%)':f"{r['saved_1r_feed_D_percent']:.5f}",
        'Product D (atom%)':f"{r['saved_1r_product_D_percent']:.5f}",
        'Enrichment factor':f"{r['saved_1r_E']:.3f}",
        'Apparent retention (%)':f"{100*vf/vi*r['saved_1r_E']:.2f}"})
for name, rows in [('stage_product_concentrations.csv', r5), ('stagewise_enrichment_inventory.csv', r6)]:
    with (out/name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
