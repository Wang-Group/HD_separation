"""Index content-named scientific assets without running analyses."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGURES = [
    ('hd_calibration_h2_control', 'mass_spectrometry/outputs/mz3_response/hd_calibration_h2_control', 'mass_spectrometry/plot_mz3_response.py', 'mass_spectrometry/inputs/mz3_response'),
    ('matched_mz3_responses', 'mass_spectrometry/outputs/mz3_response/matched_mz3_responses', 'mass_spectrometry/plot_mz3_response.py', 'mass_spectrometry/inputs/mz3_response'),
    ('nmr_calibration', 'nmr/results/figures/nmr_calibration', 'nmr/plot_response.py', 'nmr/results'),
    ('nmr_calibration_spectra_low', 'nmr/results/figures/nmr_calibration_spectra_low', 'nmr/plot_calibration_spectra.py', 'nmr/results'),
    ('nmr_calibration_spectra_high', 'nmr/results/figures/nmr_calibration_spectra_high', 'nmr/plot_calibration_spectra.py', 'nmr/results'),
    ('stagewise_deuterium_enrichment', 'nmr/results/figures/stagewise_deuterium_enrichment', 'nmr/plot_response.py', 'nmr/results'),
    ('stage_product_nmr_spectra', 'nmr/results/figures/stage_product_nmr_spectra', 'nmr/plot_response.py', 'nmr/results'),
    ('hiden_hydrogen_control', 'mass_spectrometry/outputs/interference_sensitivity/hiden_hydrogen_control', 'mass_spectrometry/plot_response.py', 'mass_spectrometry/outputs'),
    ('h3_separation_sensitivity', 'mass_spectrometry/outputs/interference_sensitivity/h3_separation_sensitivity', 'mass_spectrometry/interference_sensitivity.py', 'mass_spectrometry/outputs'),
    ('lsv_repeatability', 'LSV_repeat_analysis/figures/lsv_repeatability', 'LSV_repeat_analysis/plot_and_tables.py', 'LSV_repeat_analysis/tables'),
    ('hd_repeatability', 'HD_repeat_analysis/figures/HD_repeatability_distribution', 'HD_repeat_analysis/plot_distribution.py', 'HD_repeat_analysis/tables'),
    ('reproducibility_overview', 'reproducibility_overview/reproducibility_overview', 'reproducibility_overview/build_overview.py', 'reproducibility_overview'),
]
TABLES = [
    ('stage_product_concentrations', 'nmr/results/tables/stage_product_concentrations.csv'),
    ('stagewise_enrichment_inventory', 'nmr/results/tables/stagewise_enrichment_inventory.csv'),
]


def build_index():
    figures = []
    for name, stem, command, source in FIGURES:
        assert (ROOT/command).is_file() and (ROOT/source).is_dir(), name
        assets = {}
        for extension in ('png', 'svg', 'pdf'):
            path = ROOT/(stem+'.'+extension)
            assets[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        figures.append(dict(asset_id=name, command='python '+command, source=source, assets=assets))
    tables = [dict(asset_id=name, path=path, command='python nmr/export_tables.py',
                   sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()) for name, path in TABLES]
    return dict(naming='analysis content, independent of document numbering',
                typography_postprocess='python ../0-Figures/figure_typography.py . --apply',
                commands_working_directory='others_revision', figures=figures, tables=tables)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Validate the existing index without writing.')
    args = parser.parse_args()
    if not args.check:
        import sys
        sys.path.insert(0, str(ROOT.parent/'0-Figures'))
        from figure_typography import normalize_exports
        normalize_exports(ROOT, apply=True)
    index = build_index()
    path = ROOT/'ASSET_INDEX.json'
    if args.check:
        assert json.loads(path.read_text(encoding='utf-8')) == index, 'Refresh the asset index after exporting figures.'
    else:
        path.write_text(json.dumps(index, indent=2)+'\n', encoding='utf-8')
    print(f"{len(index['figures'])} figures and {len(index['tables'])} tables verified by content name")


if __name__ == '__main__':
    main()
