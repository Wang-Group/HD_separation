#!/usr/bin/env python3
"""Run a pyDRTtools GCV sensitivity check for Figure 3 EIS/DRT analysis."""

from __future__ import annotations

import eis_drt_hd_separation_validation as drt


def main() -> None:
    drt.OUT_DATA = drt.FIG3_DIR / "data" / "eis_pydrttools_pwl_gcv_validation"
    drt.OUT_FIG = drt.FIG3_DIR / "outputs" / "eis_pydrttools_pwl_gcv_validation"
    drt.PYDRT_RBF_TYPE = "PWL"
    drt.PYDRT_DATA_USED = "Combined Re-Im Data"
    drt.PYDRT_INDUCT_USED = 1
    drt.PYDRT_DER_USED = "1st order"
    drt.PYDRT_CV_TYPE = "GCV"
    drt.PYDRT_REG_PARAM = 1e-3
    drt.main()


if __name__ == "__main__":
    main()
