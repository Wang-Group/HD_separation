import torch
from botorch.models.gp_regression_mixed import MixedSingleTaskGP, SingleTaskGP
from botorch.models.gp_regression import HeteroskedasticSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement, qExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf_mixed, optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.models.transforms.outcome import Standardize

torch.set_default_dtype(torch.float64)


# IMPORTANT:
# 1. all the train_x should be normalized to an unit hypercube
# 2. all the train_y should be standarized

def initialize_noiseless_model(train_x, train_obj, cat_dims = None):
    """
    Create models to multi-objective optimization. If categorical values are included, use MixedSingleTaskGP
    """
    # define models for objective and constraint
    models = []
    if cat_dims == None:
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i : i + 1]
            models.append(
                SingleTaskGP(
                    train_x, train_y, outcome_transform  = Standardize(1)
                )
            )
    else:
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i : i + 1]
            models.append(
                MixedSingleTaskGP(
                    train_x, train_y, cat_dims = cat_dims
                )
            )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def initialize_heter_noise_model(train_x, train_obj, train_obj_var):
    """
    Create models to multi-objective optimization. If categorical values are included, use MixedSingleTaskGP
    """
    # define models for objective and constraint
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        train_y_var = train_obj_var[..., i : i + 1]
        models.append(
            HeteroskedasticSingleTaskGP(
                train_X = train_x, train_Y = train_y, train_Yvar = train_y_var, outcome_transform  = Standardize(1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def optimize_qnehvi_and_get_observation(model, train_x, sampler, ref_point, cat_dims = None, fixed_features_list = None):
    """Optimizes the qnoisyEHVI acquisition function, and returns a new candidate and observation."""

    # search the aquisition function in a unit hypercube by default
    standard_bounds = torch.stack([torch.zeros(train_x.shape[1]), torch.ones(train_x.shape[1])])

    if cat_dims == None:
        # partition non-dominated space into disjoint rectangles
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,  # use known reference point
            X_baseline=train_x,
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 1000},
            sequential=True,
        )
        # observe new values
        new_x = candidates.detach()
        return new_x
    
    else:
        # partition non-dominated space into disjoint rectangles
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,  # use known reference point
            X_baseline=train_x,
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf_mixed(
            acq_function = acq_func,
            bounds=standard_bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,            
            fixed_features_list=fixed_features_list)
        # observe new values
        new_x = candidates.detach()
        return new_x

def optimize_qehvi_and_get_observation(model, train_x, sampler, ref_point, cat_dims = None, fixed_features_list = None):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    partitioning = FastNondominatedPartitioning(
        ref_point = torch.tensor(ref_point),
        Y=pred
    )
    # search the aquisition function in a unit hypercube by default
    standard_bounds = torch.stack([torch.zeros(train_x.shape[1]), torch.ones(train_x.shape[1])])
    if cat_dims == None:
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 1000},
            sequential=True
        )
        # observe new values
        new_x = candidates.detach()
        return new_x
    
    else:
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler
        )
        # optimize
        candidates, _ = optimize_acqf_mixed(
            acq_function = acq_func,
            bounds=standard_bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,            
            fixed_features_list=fixed_features_list)
        # observe new values
        new_x = candidates.detach()
        return new_x

def optimize_qehvi_and_get_observation_obs(model, obs, standard_bounds, sampler, ref_point, cat_dims = None, fixed_features_list = None, inequality_constraints = None):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        partitioning = FastNondominatedPartitioning(
            ref_point = torch.tensor(ref_point),
            Y=obs
        )
    if cat_dims == None:
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=1,
            num_restarts=200,
            raw_samples=1024,  # used for intialization heuristic
            inequality_constraints = inequality_constraints
        )
        # observe new values
        new_x = candidates.detach()
        expected_y = _.detach()

        return new_x, expected_y
    
    else:
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler
        )
        # optimize
        candidates, _ = optimize_acqf_mixed(
            acq_function = acq_func,
            bounds=standard_bounds,
            q=1,
            num_restarts=200,
            raw_samples=1024,            
            fixed_features_list=fixed_features_list,
            inequality_constraints = inequality_constraints)
        # observe new values
        new_x = candidates.detach()
        expected_y = _.detach()
        return new_x, expected_y