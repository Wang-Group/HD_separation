"""Verify Hiden calibration/control calculations from the publication workbook."""
from pathlib import Path
import csv
import json
import numpy as np
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent


def verify():
    path = HERE / 'inputs/hiden_calibration_controls.xlsx'
    values = load_workbook(path, data_only=True)
    formulas = load_workbook(path, data_only=False)
    fits = {}
    for species in ['H2', 'HD']:
        sheet = values[f'{species} calibration']
        x, y = np.array([[sheet.cell(r, c).value for c in [3, 4]]
                         for r in range(32, 53)], dtype=float).T
        slope = np.sum((x-x.mean())*(y-y.mean())) / np.sum((x-x.mean())**2)
        intercept = y.mean() - slope*x.mean()
        r2 = np.corrcoef(x, y)[0, 1]**2
        fits[species] = (slope, intercept)
        np.testing.assert_allclose([sheet[f'B{i}'].value for i in range(3, 7)],
                                   [slope, intercept, r2, len(x)], rtol=1e-10, atol=1e-22)
        np.testing.assert_allclose([sheet[f'E{r}'].value for r in range(32, 53)],
                                   slope*x+intercept, rtol=1e-10, atol=1e-22)
        assert formulas[sheet.title]['B3'].value.startswith('=SLOPE(')
        assert len(formulas[sheet.title]._charts) == 1

    control, matched = values['H2 control'], values['Matched response']
    net2, net3, h2, hd_response, ratios = [], [], [], [], []
    alpha, liquid_d = matched['B3'].value, matched['B4'].value
    ratio = ((1-liquid_d)/liquid_d*alpha-1)/2
    for r in range(32, 72):
        a = control[f'C{r}'].value-control[f'D{r}'].value
        b = control[f'F{r}'].value-control[f'G{r}'].value
        concentration = (a-fits['H2'][1])/fits['H2'][0]
        hd = concentration/ratio
        response = fits['HD'][0]*hd+fits['HD'][1]
        np.testing.assert_allclose([control[f'{c}{r}'].value for c in ['E', 'H', 'I']],
                                   [a, b, concentration], rtol=1e-10, atol=1e-22)
        np.testing.assert_allclose([matched[f'{c}{r}'].value for c in ['B', 'C', 'D', 'E', 'F']],
                                   [concentration, hd, response, b, b/response], rtol=1e-10, atol=1e-22)
        net2.append(a); net3.append(b); h2.append(concentration)
        hd_response.append(response); ratios.append(b/response)
    x, y = np.asarray(h2), np.asarray(net3)
    slope = np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)
    intercept = y.mean()-slope*x.mean()
    np.testing.assert_allclose([control['B3'].value, control['B4'].value, control['B5'].value],
                               [slope, intercept, np.corrcoef(x, y)[0, 1]**2], rtol=1e-10, atol=1e-22)
    with (HERE/'outputs/hiden_pulse_windows.csv').open(encoding='utf-8-sig', newline='') as stream:
        existing = list(csv.DictReader(stream))
    np.testing.assert_allclose(net3, [float(r['net_m3_mbar']) for r in existing], rtol=1e-10, atol=0)
    assert sum(len(s._charts) for s in formulas) == 4
    assert [s.title for s in formulas] == ['H2 calibration', 'HD calibration', 'H2 control', 'Matched response']
    return {'analysis': 'Hiden numerical source workbook', 'calibration_readings': 42,
            'control_plateaus': 40, 'native_charts': 4, 'cached_formulas_match': True,
            'signal_ratio_percent_range': [100*min(ratios), 100*max(ratios)]}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
