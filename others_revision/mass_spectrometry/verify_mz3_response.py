"""Validate the adopted response-curve tables against the numerical workbook."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent


def verify():
    source = HERE / 'inputs/mz3_response'
    parameters = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    calibration = pd.read_csv(source / 'calibration_points.csv')
    points = pd.read_csv(source / 'matched_response_points.csv')
    curves = pd.read_csv(source / 'response_curves.csv')
    workbook = load_workbook(HERE / 'inputs/hiden_calibration_controls.xlsx', data_only=True)
    fits = {}
    for species in ['H2', 'HD']:
        values = calibration.loc[calibration.species.eq(species)]
        assert len(values) == 21
        sheet = workbook[f'{species} calibration']
        expected = np.array([[sheet.cell(r, c).value for c in [3, 4]] for r in range(32, 53)])
        actual = values[['concentration_volpct', 'baseline_corrected_signal_mbar']].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-22)
        fit = parameters['calibration_reproduction'][species]
        np.testing.assert_allclose([fit['slope'], fit['intercept'], fit['R2'], fit['n']],
                                   [sheet[f'B{r}'].value for r in range(3, 7)], rtol=1e-10, atol=1e-22)
        fits[species] = fit
    assert len(points) == 40
    sheet = workbook['Matched response']
    columns = ['h2_concentration_volpct', 'hd_concentration_volpct',
               'mz3_hd_calibration_total_mbar', 'mz3_h2_net_mbar']
    expected = np.array([[sheet.cell(r, c).value for c in [2, 3, 4, 5]] for r in range(32, 72)])
    np.testing.assert_allclose(points[columns].to_numpy(), expected, rtol=1e-10, atol=1e-22)
    alpha, liquid_d = parameters['alpha_assumed'], parameters['liquid_D_atom_fraction']
    ratio = (alpha * (1 - liquid_d) / liquid_d - 1) / 2
    np.testing.assert_allclose(ratio, parameters['H2_to_HD_molecular_concentration_ratio'])
    control_fit = parameters['control_descriptive_fits'][0]
    assert control_fit['degree'] == parameters['plotted_control_fit_degree'] == 1
    slope, intercept = control_fit['coefficients_descending_mbar_per_volpct_power']
    sheet = workbook['H2 control']
    np.testing.assert_allclose([slope, intercept, control_fit['descriptive_R2']],
                               [sheet[f'B{r}'].value for r in [3, 4, 5]], rtol=1e-10, atol=1e-22)
    np.testing.assert_allclose(curves.h2_concentration_volpct, ratio * curves.hd_concentration_volpct,
                               rtol=1e-10, atol=0)
    np.testing.assert_allclose(curves.mz3_hd_calibration_total_mbar,
                               fits['HD']['slope'] * curves.hd_concentration_volpct + fits['HD']['intercept'],
                               rtol=1e-10, atol=1e-22)
    np.testing.assert_allclose(curves.mz3_h2_linear_fit_net_mbar,
                               slope * curves.h2_concentration_volpct + intercept, rtol=1e-10, atol=1e-22)
    np.testing.assert_allclose(curves.h2_to_hd_signal_percent,
                               100 * curves.mz3_h2_linear_fit_net_mbar / curves.mz3_hd_calibration_total_mbar,
                               rtol=1e-10, atol=1e-12)
    for name in ['hd_calibration_h2_control', 'matched_mz3_responses']:
        for extension in ['svg', 'png', 'pdf']:
            assert (HERE / 'outputs/mz3_response' / f'{name}.{extension}').is_file()
    return {'analysis': 'Matched m/z = 3 response', 'calibration_readings': 42,
            'controls': 40, 'workbook_match': True, 'fits_changed': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
