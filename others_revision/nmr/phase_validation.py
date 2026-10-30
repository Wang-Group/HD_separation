"""Pure functions for the non-adopted ACME robustness check."""
import numpy as np
import nmrglue as ng
from scipy.optimize import minimize_scalar

def peak(x, y, lo, hi):
    ii = np.flatnonzero((x >= lo) & (x <= hi))
    i = int(ii[np.argmax(y[ii])])
    offset = 0.
    if 0 < i < len(y)-1:
        den = y[i-1]-2*y[i]+y[i+1]
        if den < 0:
            offset = float(np.clip(.5*(y[i-1]-y[i+1])/den, -.5, .5))
    return float(x[i]+offset*(x[1]-x[0]))

def baseline(x, y, c, h=.25):
    a = (x >= c-h-.15) & (x <= c-h-.05)
    b = (x >= c+h+.05) & (x <= c+h+.15)
    assert a.sum() > 10 and b.sum() > 10
    return np.interp(x, [np.median(x[a]), np.median(x[b])],
                     [np.median(y[a]), np.median(y[b])])

def integrate(x, y, c, h=.25):
    m = np.abs(x-c) <= h
    return float(np.trapezoid((y-baseline(x, y, c, h))[m], x[m]))

def local_phase(x, z, c, h):
    m = np.abs(x-c) <= h
    cb = baseline(x, z.real, c, h) + 1j*baseline(x, z.imag, c, h)
    zz = (z-cb)[m]
    zz /= np.max(np.abs(zz))
    initial = float(-np.angle(np.trapezoid(zz, x[m]), deg=True))
    def score(angle):
        yy = (zz*np.exp(1j*np.deg2rad(angle))).real
        # ACME can spuriously favor an all-negative spectrum because of its
        # signed max denominator. Enforce the physical positive-peak branch.
        if yy.sum() <= 0 or yy.max() <= 0:
            return 1e6
        return ng.proc_autophase._ps_acme_score([float(angle), 0.], zz)
    grid = np.linspace(initial-60, initial+60, 121)
    scores = np.array([score(v) for v in grid])
    i = int(np.argmin(scores))
    lo, hi = max(grid[0], grid[i]-2), min(grid[-1], grid[i]+2)
    res = minimize_scalar(score, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
    assert res.success and score(res.x) < 1e6
    return float(res.x), bool(abs(res.x-initial) > 58)

def phase_and_measure(x, z, d0, w0, h=.25):
    pd, edge_d = local_phase(x, z, d0, h)
    pw, edge_w = local_phase(x, z, w0, h)
    anchors = np.rad2deg(np.unwrap(np.deg2rad([pd, pw])))
    slope = float((anchors[1]-anchors[0])/(w0-d0))
    phi = anchors[0]+slope*(x-d0)
    y = (z*np.exp(1j*np.deg2rad(phi))).real
    dc = peak(x, y, d0-.08, d0+.08)
    wc = peak(x, y, w0-.08, w0+.08)
    darea, warea = integrate(x, y, dc), integrate(x, y, wc)
    assert darea > 0 and warea > 0
    return {"y": y, "d": dc, "w": wc, "darea": darea, "warea": warea,
            "ratio": warea/darea, "phase_ref_deg": float(anchors[0]),
            "phase_water_deg": float(anchors[1]), "phase_slope_deg_ppm": slope,
            "phase_search_boundary": edge_d or edge_w}
