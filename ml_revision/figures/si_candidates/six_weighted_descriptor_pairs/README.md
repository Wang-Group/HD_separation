# Six weighted descriptor pairs for SI review

Run `python plot_six_weighted_descriptor_pairs.py` from ml_revision. Fifteen pairs, each with precursor/global, ICP/global, precursor/local and ICP/local views. Arial; original discrete color palette and marginal histograms.

Pairwise matched observed data, not an all-six complete-case restriction. Initial H/D outcomes are original, not repeated averages. BO records are independent overlays only. See manifest.json for weighting, assumptions, fixed neighbor normalization and display-dependent masking. Local limits cover both representations of all available BO observations, plus 5.5% padding; these newly defined limits are not claimed to reproduce the three-property map limits. Values and units are inherited from the packaged elemental property table; weighted frequency is a descriptor, not a measured alloy spectrum. No ML refit or response/SI edit.

Each pair has PNG/SVG/PDF and four compressed surface grids. all_15_pairs.pdf collects the figures. CSVs preserve weighted values, memberships and panel counts.
