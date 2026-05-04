# LSV Geometry-Area Notes

Most LSV files in this folder follow the standard geometry convention used for
the initial-screening and BO datasets.

Three initial-screening catalysts were measured using a 2 cm^2 geometric area:

- Ag-Ce
- Co-V
- Mo-Mg

For these three samples, the LSV current axis should be interpreted with the
2 cm^2 geometry in mind. When extracting the value comparable to eta10 under
the standard 1 cm^2 convention, use the raw potential at -20 mA from the
exported LSV curve.

In other words, for these three samples:

`corrected eta10 = raw eta20`

The file `lsv_geometry_area_metadata.csv` records this exception in
machine-readable form.
