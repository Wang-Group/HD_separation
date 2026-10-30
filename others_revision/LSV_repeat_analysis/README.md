# LSV Repeatability Analysis

Updated 25 October 2026 from the current automatic LSV extraction, including
the single documented Co-Ce repeat4 noise exception.

## Current results

- 1,583 curves retain their extraction records and 677 source sample labels.
- Pooling the five audited same-recipe series gives 672 plotted catalysts.
- 1,105 accepted crossings: 1,030 measurements in 349 catalysts with SD,
  plus 75 catalysts with one crossing. The other 248 have no measured eta10.
- Median SD 24.01 mV; mean SD 31.32 mV; 95th percentile 85.14 mV.
- 221/349 (63.32%) have SD <=30 mV; 286/349 (81.95%) have SD <=50 mV.
- Eleven catalysts have SD >100 mV. None were removed to narrow the distribution.

Means and sample SDs use uncapped crossing values and the n-1 denominator.
Rising/nearest estimates, scan limits and unassigned results never enter eta10
means or SDs. This is across-measurement variability, not SEM or proof of
independently prepared triplicates. Acquisition conditions may differ.

## Files and reproduction

Run from this directory:

```bash
python plot_and_tables.py
```

This calls the shared plotting analysis in
`../../0-Figures/Figure1/revision_figure/plot_eta10_repeatability.py`.
Its aggregation module `lsv_plot_data.py` is also used by Figure 1B and the
joint reproducibility overview. It rebuilds means/SDs from curve records and
validates them against the extraction before the same-recipe pooling.

- `figures/lsv_repeatability.{pdf,png,svg}`: current Arial histogram
  and empirical cumulative distribution.
- `tables/plot_sample_statistics.csv`: current pooled plotting statistics.
- `tables/sample_SD_distribution.csv`: the 349 measured-eta10 SDs.
- `tables/replicate_counts.csv`, `SD_thresholds.csv`, `summary.json`: counts
  and numerical results.
- `data/eta10_*.csv`: unchanged-identity extraction snapshots, before pooling.
- `source_manifest.json`: source paths and hashes.
- `SI_MS_integration_notes.md`: proposed SI/MS integration using these results.

The current figure and statistics support LSV repeatability in Reviewer 1 Comment 8.
Raw TXT, extraction selections, ML and Figure 4 are unchanged. The response
integration uses tracked changes; it does not replace the working SI/manuscript.
