"""Training-only correlation groups and resampled permutation ranking."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from model_core import estimator
from descriptors import DescriptorTransform
from workflow.recipes import RecipeGuard


def correlation_groups(x, threshold):
    if not 0 < threshold <= 1:
        raise ValueError('Invalid correlation threshold')
    corr = pd.DataFrame(x).corr(method='spearman').fillna(0).to_numpy()
    corr = np.clip((corr + corr.T) / 2, -1, 1)
    np.fill_diagonal(corr, 1)
    distance = 1 - np.abs(corr)
    labels = fcluster(linkage(squareform(distance, checks=True), method='complete'),
                      1 - threshold + 1e-12, criterion='distance')
    groups = [np.flatnonzero(labels == k).tolist() for k in np.unique(labels)]
    groups.sort(key=lambda group: min(group))
    for group in groups:
        assert np.min(np.abs(corr[np.ix_(group, group)])) >= threshold - 1e-10
    return groups, corr


def permutation_utilities(model, x, y, groups, repeats, seed):
    """Jointly permute a group's columns using the same row permutation."""
    rng = np.random.default_rng(seed)
    batches = [x]
    for group in groups:
        for _ in range(repeats):
            p = x.copy()
            p[:, group] = x[rng.permutation(len(x))][:, group]
            batches.append(p)
    predicted = model.predict(np.concatenate(batches)).reshape(len(batches), len(x))
    mse = np.mean((predicted - np.asarray(y)[None, :]) ** 2, axis=1)
    scale = max(float(np.var(y)), 1e-12)
    values = (mse[1:].reshape(len(groups), repeats) - mse[0]) / scale
    return values.mean(axis=1), values.std(axis=1, ddof=1) if repeats > 1 else np.zeros(len(groups))


def resample_splits(frame, protocol, count, fraction, seed):
    guard = RecipeGuard(frame)
    if protocol == 'unseen_pair':
        split = GroupShuffleSplit(n_splits=count, train_size=fraction, random_state=seed)
        iterator = split.split(frame, groups=frame.metal_pair)
    elif guard.pairs:
        parents = {s: s for s in frame['sample']}

        def root(s):
            while parents[s] != s:
                s = parents[s]
            return s

        for a, b, _ in guard.pairs:
            parents[root(a)] = root(b)
        groups = [root(s) for s in frame['sample']]
        split = GroupShuffleSplit(n_splits=count, train_size=fraction, random_state=seed)
        iterator = split.split(frame, groups=groups)
    else:
        bins = pd.qcut(frame.target, 5, labels=False, duplicates='drop')
        split = StratifiedShuffleSplit(n_splits=count, train_size=fraction, random_state=seed)
        iterator = split.split(frame, bins)
    for a, b in iterator:
        train, valid = frame.iloc[a], frame.iloc[b]
        guard.check(train, valid, 'selector resample')
        assert not set(train['sample']) & set(valid['sample'])
        if protocol == 'unseen_pair':
            assert not set(train.metal_pair) & set(valid.metal_pair)
        yield train, valid


def fit_selector(frame, protocol, threshold, config, pilot, seed, scheme):
    transform = DescriptorTransform(scheme).fit(frame)
    features = transform.features
    groups, corr = correlation_groups(transform.transform(frame).to_numpy(), threshold)
    rng = np.random.default_rng(seed + 181)
    cycles = [rng.permutation(group).tolist() for group in groups]
    trials, membership = [], []
    # Clusters belong to this fitting partition. Internal resamples estimate ranking
    # stability; their scores are not reported as independent predictive validation.
    for repeat, (train, valid) in enumerate(resample_splits(
            frame, protocol, pilot['resamples'], pilot['resample_train_fraction'], seed + 271)):
        transform_r = DescriptorTransform(scheme).fit(train)
        xt, xv = (transform_r.transform(part).to_numpy() for part in (train, valid))
        membership.append(dict(resample=repeat, train=train['sample'].tolist(),
                               validation=valid['sample'].tolist()))
        for mi, model_name in enumerate(config['models']):
            trial = repeat * len(config['models']) + mi
            mask = [cycle[trial % len(cycle)] for cycle in cycles]
            model = estimator(model_name, config['path_config'], config['selection_trees'],
                              seed + 1000 + trial).fit(xt[:, mask], train.target)
            values, _ = permutation_utilities(model, xv[:, mask], valid.target,
                [[j] for j in range(len(mask))], pilot['ranking_permutations'], seed + 3000 + trial)
            trials.append(dict(resample=repeat, model=model_name, mask=mask,
                               utilities=values.tolist()))
    utilities = np.array([trial['utilities'] for trial in trials])
    group_means = utilities.mean(axis=0)
    group_order = sorted(range(len(groups)), key=lambda g: (-group_means[g], min(groups[g])))
    representatives, alternatives = [], []
    for gi, group in enumerate(groups):
        members = []
        for index in group:
            values = [trial['utilities'][gi] for trial in trials if trial['mask'][gi] == index]
            assert values, 'Every candidate representative must be assessed'
            members.append(dict(feature=features[index], index=index, trials=len(values),
                                mean=float(np.mean(values)), sd=float(np.std(values, ddof=1))))
        winner = sorted(members, key=lambda r: (-r['mean'], r['index']))[0]
        representatives.append(winner['index'])
        alternatives.append(members)
    ranked = [representatives[gi] for gi in group_order]
    assert len(ranked) == len(set(ranked)) == len(groups)
    return dict(groups=groups, correlation=corr.tolist(), group_order=group_order,
                representatives=representatives, ranked=ranked,
                alternatives=alternatives, group_mean_utilities=group_means.tolist(),
                trials=trials, resample_memberships=membership,
                fitting_samples=frame['sample'].tolist(), medians=transform.medians)


def choose_candidates(rows, folds):
    ordered = sorted(rows, key=lambda r: (-r['mean'], r['k'], r['model'], r['config_id']))
    best = ordered[0]
    threshold = best['mean'] - best['sd'] / np.sqrt(folds)
    eligible = [r for r in rows if r['mean'] >= threshold - 1e-12]
    small = min(r['k'] for r in eligible)
    compact = sorted([r for r in eligible if r['k'] == small],
                     key=lambda r: (-r['mean'], r['model'], r['config_id']))[0]
    return best, compact, float(threshold)
