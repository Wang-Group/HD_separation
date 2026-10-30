"""Seeded estimators and atomic run records."""
import hashlib
import json
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


def estimator(name, parameters, trees, seed):
    cls = {'RF': RandomForestRegressor, 'ET': ExtraTreesRegressor}[name]
    return cls(n_estimators=trees, random_state=seed, n_jobs=1, **parameters)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
