"""Shared deterministic LSV selection, with explicit reviewed noise exceptions."""
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

TARGET = -10.0
CONFIG = dict(grid_step_mV=1, median_window_points=5, prominence_mA_cm2=.5,
              peak_distance_mV=10, left_margin_mV=5, right_margin_mV=30,
              recovery_window_mV=100, minimum_recovery_mA_cm2=2,
              minimum_curve_span_mV=20, maximum_reversal_mV=1)


def crossings(eta, j):
    result = []
    i = 0
    while i < len(j):
        if j[i] == TARGET:
            end = i
            while end + 1 < len(j) and j[end + 1] == TARGET:
                end += 1
            k = max(range(i, end + 1), key=lambda k: eta[k])
            result.append((eta[k], k, k))
            i = end + 1
            continue
        if i + 1 < len(j) and (j[i] - TARGET) * (j[i + 1] - TARGET) < 0:
            fraction = (TARGET - j[i]) / (j[i + 1] - j[i])
            result.append((eta[i] + fraction * (eta[i + 1] - eta[i]), i, i + 1))
        i += 1
    return result


def select(eta, j):
    hits = crossings(eta, j)
    if j[-1] > TARGET:
        k = max(range(len(eta)), key=lambda k: eta[k])
        return 'scan_limit', eta[k], j[k], k, k, hits
    if hits:
        value, before, after = max(hits, key=lambda p: p[0])
        return 'eta10_crossing', value, TARGET, before, after, hits
    if all(v < TARGET for v in j):
        return 'above_target_no_crossing', None, None, None, None, hits
    return 'needs_review', None, None, None, None, hits


def rising_candidates(eta, j):
    """Detect on a smoothed grid, but retain coordinates from the original points."""
    eta, j = np.asarray(eta), np.asarray(j)
    if np.any(np.diff(eta) < -CONFIG['maximum_reversal_mV']):
        return []
    if eta[-1] <= eta[0] or eta.max() - eta[0] < CONFIG['minimum_curve_span_mV']:
        return []
    unique_eta, first = np.unique(eta, return_index=True)
    grid = np.arange(np.ceil(unique_eta.min()), np.floor(unique_eta.max()) + 1,
                     CONFIG['grid_step_mV'])
    if len(grid) < 21:
        return []
    magnitude = median_filter(np.interp(grid, unique_eta, -j[first]),
                              size=CONFIG['median_window_points'], mode='nearest')
    minima, properties = find_peaks(-magnitude, prominence=CONFIG['prominence_mA_cm2'],
                                    distance=CONFIG['peak_distance_mV'])
    result = []
    for position, k in enumerate(minima):
        if k < CONFIG['left_margin_mV'] or k + CONFIG['right_margin_mV'] >= len(grid):
            continue
        end = min(k + CONFIG['recovery_window_mV'], len(grid) - 1)
        recovery = float(np.median(magnitude[max(k + 10, end - 10):end + 1]) - magnitude[k])
        if recovery < CONFIG['minimum_recovery_mA_cm2']:
            continue
        raw_index = int(np.argmin(abs(eta - grid[k])))
        result.append(dict(index=raw_index, potential_mV=float(eta[raw_index]),
            j_mA_cm2=float(j[raw_index]), grid_potential_mV=float(grid[k]),
            prominence_mA_cm2=float(properties['prominences'][position]),
            recovery_mA_cm2=recovery))
    return result


def choose_latest(candidates):
    eligible = [c for c in candidates if -20 < c['j_mA_cm2'] < -10]
    return max(eligible, key=lambda c: c['potential_mV']) if eligible else None


def automatic_selection(eta, j):
    status, value, current, before, after, hits = select(eta, j)
    base_status = status
    last_crossing = max((h[0] for h in hits), default=None)
    candidates = []
    if status == 'scan_limit':
        diagnostic = 'terminal_scan_limit_priority'
    else:
        if any(eta[k + 1] < eta[k] - CONFIG['maximum_reversal_mV'] for k in range(len(eta) - 1)):
            diagnostic = 'reversal_exceeds_1mV_no_rising_estimate'
        elif eta[-1] <= eta[0] or max(eta) - eta[0] < CONFIG['minimum_curve_span_mV']:
            diagnostic = 'insufficient_forward_span_no_rising_estimate'
        else:
            diagnostic = 'detector_applied'
        for candidate in rising_candidates(eta, j):
            in_window = -20 < candidate['j_mA_cm2'] < -10
            later = last_crossing is None or candidate['potential_mV'] > last_crossing
            candidates.append(dict(candidate, within_current_window=in_window,
                after_last_crossing=later, eligible=in_window and later, selected=False))
        eligible = [c for c in candidates if c['eligible']]
        if eligible:
            chosen = max(eligible, key=lambda c: c['potential_mV'])
            chosen['selected'] = True
            status, value, current = 'rising_point', chosen['potential_mV'], chosen['j_mA_cm2']
            before = after = chosen['index']
        elif status == 'above_target_no_crossing':
            k = min(range(len(j)), key=lambda k: (abs(j[k] + 10), -eta[k]))
            if -20 < j[k] < -10:
                status, value, current = 'nearest_point_estimate', eta[k], j[k]
                before = after = k
    return dict(status=status, potential_mV=value, j_mA_cm2=current, before=before, after=after,
        base_status=base_status, largest_crossing_mV=last_crossing, crossing_count=len(hits),
        detector_diagnostic=diagnostic, candidates=candidates)


def apply_noise_exception(decision, eta, j, exception):
    """Reject a reviewed noise-driven rise without bypassing the terminal gate."""
    assert exception['action'] == 'retain_largest_crossing_noise'
    assert decision['status'] == 'rising_point', 'Exception requires a new review if automatic selection changes'
    assert abs(decision['potential_mV'] - exception['rejected_automatic_potential_mV']) < 1e-6
    assert abs(decision['j_mA_cm2'] - exception['rejected_automatic_j_mA_cm2']) < 1e-6
    status, value, current, before, after, _ = select(eta, j)
    assert status == 'eta10_crossing', 'Noise exception cannot override a scan limit or invent a crossing'
    assert abs(value - exception['retained_crossing_mV']) < 1e-6
    return dict(decision, status=status, potential_mV=value, j_mA_cm2=current, before=before, after=after)
