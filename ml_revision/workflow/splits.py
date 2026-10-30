"""Two validation protocols; initial samples are the held-out observations."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from .recipes import RecipeGuard

PROTOCOLS = ('stratified_random', 'unseen_pair')


def partitions(frame, protocol, folds, seed, recipe_guard=None, context='partition'):
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unsupported protocol: {protocol}')
    if recipe_guard is None:
        recipe_guard = RecipeGuard(frame)
    bins = pd.qcut(frame.target, 5, labels=False, duplicates='drop')
    if protocol == 'stratified_random':
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        result = cv.split(frame, bins)
    else:
        cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
        result = cv.split(frame, bins, frame.metal_pair)
    for a, b in result:
        training, validation = frame.iloc[a].copy(), frame.iloc[b].copy()
        check(training, validation, protocol, recipe_guard, context)
        yield training, validation


def eligible_bo(bo, training, validation, protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(protocol)
    if protocol == 'stratified_random':
        return bo.loc[~bo.composition_group.isin(validation.composition_group)].copy()
    return bo.loc[bo.metal_pair.isin(training.metal_pair)].copy()


def check(training, validation, protocol, recipe_guard=None, context='partition'):
    assert protocol in PROTOCOLS
    assert training['sample'].is_unique and validation['sample'].is_unique
    assert not set(training['sample']) & set(validation['sample'])
    assert not set(training.icp_group) & set(validation.icp_group)
    if protocol == 'unseen_pair':
        assert not set(training.metal_pair) & set(validation.metal_pair)
    bo = training.loc[training.cohort.eq('BO')]
    assert not set(bo.composition_group) & set(validation.composition_group)
    assert np.isfinite(training.target).all() and np.isfinite(validation.target).all()
    if recipe_guard is None:
        recipe_guard = RecipeGuard(pd.concat([training, validation]))
    recipe_guard.check(training, validation, context)


def design(data, config, include_bo=True):
    recipe_guard = RecipeGuard(data)
    initial = data.loc[data.cohort.eq('initial')].copy()
    bo = data.loc[data.cohort.eq('BO')].copy()
    jobs = []
    for protocol in config['protocols']:
        for repeat in range(config['repeats']):
            for fold, (base, test) in enumerate(partitions(initial, protocol, config['outer_folds'], 42+repeat,
                                                          recipe_guard, 'outer')):
                for arm in (('without_bo','with_bo') if include_bo else ('without_bo',)):
                    extra = eligible_bo(bo, base, test, protocol) if arm == 'with_bo' else bo.iloc[:0]
                    train = pd.concat([base, extra])
                    check(train, test, protocol, recipe_guard, 'outer')
                    seed = (7000 if protocol == 'stratified_random' else 2000) + repeat*100 + fold
                    inner = []
                    for a, b in partitions(base, protocol, config['inner_folds'], seed, recipe_guard, 'inner'):
                        add = eligible_bo(extra, a, b, protocol)
                        a = pd.concat([a, add])
                        check(a, b, protocol, recipe_guard, 'inner')
                        assert not (set(a['sample'])|set(b['sample'])) & set(test['sample'])
                        inner.append((a,b))
                    jobs.append(dict(protocol=protocol,repeat=repeat,fold=fold,arm=arm,
                                     train=train,test=test,inner=inner,seed=seed))
    return jobs
