"""Reproduce and package the shared Figure 1B / LSV repeatability statistics."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGURE = REPO / '0-Figures/Figure1/revision_figure'
sys.path.insert(0, str(FIGURE))
from plot_eta10_repeatability import main as plot_repeatability


def main():
    plot_repeatability()
    review = FIGURE / 'repeatability_review'
    files = {
        FIGURE / 'eta10_curves.csv': HERE / 'data/eta10_curves.csv',
        FIGURE / 'eta10_samples.csv': HERE / 'data/eta10_samples.csv',
        review / 'plot_sample_statistics.csv': HERE / 'tables/plot_sample_statistics.csv',
        review / 'summary.json': HERE / 'tables/summary.json',
        review / 'thresholds.csv': HERE / 'tables/SD_thresholds.csv',
        review / 'replicate_counts.csv': HERE / 'tables/replicate_counts.csv',
        review / 'sample_SD_distribution.csv': HERE / 'tables/sample_SD_distribution.csv',
        review / 'largest_20_SDs.csv': HERE / 'tables/largest_20_SDs.csv',
    }
    for extension in ['pdf','png','svg']:
        files[review / ('eta10_SD_distribution.'+extension)] = HERE / 'figures' / ('lsv_repeatability.'+extension)
    snapshots = []
    for source, target in files.items():
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        snapshots.append(dict(source=source.relative_to(REPO).as_posix(),
            local=target.relative_to(HERE).as_posix(), sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    shutil.copyfile(review / 'summary.json', HERE / 'data/existing_SD_summary.json')
    manifest = dict(rule='2026-10-25-automatic-rising-with-noise-review', snapshots=snapshots,
        aggregation='Shared lsv_plot_data.py; audited same-recipe measurements pooled from individual curves.',
        source_validation=json.loads((review/'verification.json').read_text(encoding='utf-8')),
        SI_draft_status='Current tables and figures support response LSV repeatability; SI/MS integration is proposed in SI_MS_integration_notes.md.')
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
