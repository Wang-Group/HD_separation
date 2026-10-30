"""Fail-closed recipe-overlap checks; never change fold assignments or inputs."""
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd

REFERENCE = Path(__file__).resolve().parents[1] / 'inputs/reference'
TOLERANCE = 1e-9


def full_recipe(row):
    amounts = {}
    for i in range(1, 5):
        name, volume = row.get(f'substance{i}'), row.get(f'volume{i}')
        if pd.isna(name) and pd.isna(volume):
            continue
        if not isinstance(name, str) or not name.strip() or pd.isna(volume):
            raise ValueError('Incomplete precursor identity/volume record')
        value = float(volume)
        if not np.isfinite(value) or value < 0:
            raise ValueError('Recipe volumes must be finite and nonnegative')
        if value:
            amounts[name] = amounts.get(name, 0.) + value
    return tuple(sorted(amounts.items())) if amounts else None


def packaged_recipes():
    four = pd.read_csv(REFERENCE / 'BO_recipes.csv')
    four['sample'] = four.system + four.row_id.astype(str)
    cf = pd.DataFrame(json.loads((REFERENCE / 'CoFe_recipes.json').read_text(encoding='utf-8')))
    cf['sample'] = 'Co-Fe' + cf.index.astype(str)
    rows = pd.concat([four, cf]).set_index('sample', verify_integrity=True)
    return {sample: full_recipe(row) for sample, row in rows.iterrows()}


def same_full_recipe(a, b):
    return (tuple(k for k, _ in a) == tuple(k for k, _ in b) and
            np.allclose([v for _, v in a], [v for _, v in b], atol=TOLERANCE, rtol=0))


def metal_signature(row):
    a, b = row['element1'], row['element2']
    if not all(isinstance(m, str) and m.strip() for m in (a, b)):
        raise ValueError(f"Missing metal identities for {row['sample']}")
    fraction = float(row['nominal_frac_element1'])
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError(f"Invalid nominal precursor fraction for {row['sample']}")
    amounts = {}
    for metal, value in [(a, fraction), (b, 1 - fraction)]:
        if value:
            amounts[metal] = amounts.get(metal, 0.) + value
    metals = tuple(sorted(amounts))
    return metals, amounts[metals[0]]


class RecipeGuard:
    """Resolve possible duplicates once, then check every nested boundary."""

    def __init__(self, frame, recipes=None):
        required = {'sample', 'element1', 'element2', 'nominal_frac_element1'}
        if not required <= set(frame.columns):
            raise ValueError('Recipe checks need sample, metal identities and nominal precursor fraction')
        if not frame['sample'].is_unique or frame['sample'].isna().any():
            raise ValueError('Recipe checks need unique, nonempty sample IDs')
        recipes = packaged_recipes() if recipes is None else recipes
        self.samples = set(frame['sample'])
        groups = {}
        for row in frame.to_dict('records'):
            sample = row['sample']
            metals, fraction = metal_signature(row)
            explicit, supplied = full_recipe(row), recipes.get(sample)
            if explicit is not None and supplied is not None and not same_full_recipe(explicit, supplied):
                raise ValueError(f'Conflicting full recipe records for {sample}')
            recipe = explicit if explicit is not None else supplied
            if recipe is not None:
                amounts = dict(recipe)
                total = sum(amounts.get(m, 0.) for m in set([row['element1'], row['element2']]))
                if total <= 0 or not np.isclose(amounts.get(metals[0], 0.) / total,
                                                fraction, atol=TOLERANCE, rtol=0):
                    raise ValueError(f'Full recipe disagrees with nominal metal ratio for {sample}')
            groups.setdefault(metals, []).append((sample, fraction, recipe))
        self.pairs = []
        for records in groups.values():
            for (a, fa, ra), (b, fb, rb) in combinations(records, 2):
                if not np.isclose(fa, fb, atol=TOLERANCE, rtol=0):
                    continue
                if ra is None or rb is None:
                    self.pairs.append((a, b, 'unresolved same-recipe candidate'))
                elif same_full_recipe(ra, rb):
                    self.pairs.append((a, b, 'same recipe'))

    def check(self, training, validation, context='partition'):
        train, test = set(training['sample']), set(validation['sample'])
        if not (train | test) <= self.samples:
            raise ValueError('Fold contains samples outside the recipe-check scope')
        overlaps = [(a, b, reason) for a, b, reason in self.pairs
                    if (a in train and b in test) or (b in train and a in test)]
        if overlaps:
            detail = '; '.join(f'{a} / {b}: {reason}' for a, b, reason in overlaps)
            raise AssertionError(f'Recipe overlap across {context} train/held-out boundary: {detail}')
