"""BoTorch active-learning helpers for the electrochemical HD platform.

Paper-relevant BO path:
    platform_manager_HD_Hcell.py
        -> generate_new_exp_MO_one_system_V2(...)

`generate_new_exp_MO_one_system_V2` is the four-component bimetallic
recipe optimizer used for the HD separation workflow. It optimizes three
composition variables (metal 1, metal 2, and sodium citrate volumes) with
two objectives:

    1. HER activity, represented by lsv_overpotential_-0.01 after a
       positive shift so that better activity is maximized.
    2. H/D separation coefficient.

The acquisition function is BoTorch's
qNoisyExpectedHypervolumeImprovement, i.e. qNEHVI.

Older single-objective and exploratory helper functions are retained below
because they document earlier development paths, but they should not be
cited as the Figure 4 BO algorithm unless explicitly used by a platform
manager or notebook.

Canonical location:
    EC_tool_sys/botorch_test/botorch_tool/active_learning.py
"""

import torch
from botorch import fit_gpytorch_mll
import botorch_tool
from botorch.sampling.normal import SobolQMCNormalSampler
import numpy as np
import pandas as pd
import pandas as pd
from botorch.optim import optimize_acqf
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

torch.set_default_dtype(torch.float64)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

repeat_exp = pd.Series(
    {
        "substance1": "Co",
        "volume1": 4.0,
        "substance2": "Cd",
        "volume2": 4.0,
        "substance3": "Na3Cit",
        "volume3": 3.0,
        "substance4": "H3BO3",
        "volume4": 4.0,
        "substance5": "water",
        "volume5": 5.0,
        "syn_completed": "None",
        "storage_location": "None",
        "in_storage": "None",
        "cat_completed": "None",
        "lsv_overpotential_-0.01": "None",
        "lsv_overpotential_-0.02": "None",
        "lsv_overpotential_-0.03": "None",
        "lsv_overpotential_-0.04": "None",
        "lsv_overpotential_-0.05": "None",
    }
)
repeat_exp = repeat_exp.replace({np.nan: None, "None": None})


# def generate_new_exp_SO(exp_data: pd.DataFrame, exp_index: int):
#     """
#     Generate a series of new experiments from the previous data and the exp_index for a single objective optimization
#     Args:
#         exp_data: a pandas.dataframe that records the experiments
#         exp_index: the index of the experiments when we call generate_new_exp, which determines the categories of the reactants
#     Returns:
#         the next sampling points or None
#     """
#     # set up the random seed
#     np.random.seed(42)
#     torch.manual_seed(42)

#     # get the current combo of the experiments
#     current_combo = exp_data[["substance1", "substance2"]].iloc[exp_index]
#     identical_index = (exp_data[["substance1", "substance2"]] == current_combo).all(
#         axis=1
#     )
#     selected_index = np.where(identical_index == True)[0]

#     # check if all the experiemnts are characterized for this specific combo
#     if sum(exp_data.loc[identical_index]["lsv_overpotential_-0.02"].values != None) == sum(identical_index):
#         # read in the data
#         x = []
#         y_mean = []
#         y_var = []
#         # get the selected dataset
#         for j in selected_index:
#             # append the x value first
#             x_temp = [
#                 (exp_data.iloc[j]["volume1"] - 0.5) / 9.5,
#                 (exp_data.iloc[j]["volume2"] - 0.5) / 9.5,
#                 (exp_data.iloc[j]["volume3"] / 5),
#                 (exp_data.iloc[j]["volume4"] / 5),
#             ]
#             x.append(x_temp)
#             # We assign a normal value to it if the results are not "unknown"
#             if exp_data.iloc[j]["lsv_overpotential_-0.02"] != "unknown":
#                 y_mean.append(1 * exp_data.iloc[j]["lsv_overpotential_-0.02"])
                
#             # if the results are unknown, it is a bad point and we should assign a very low value to it
#             else:
#                 y_mean.append(-1.30)
#             # append a 0.03 V std to the list
#             y_var.append((0.03**2))

#         # process the data input numpy array
#         x = np.array(x).reshape(-1, 4)
#         y_mean = np.array(y_mean).reshape(-1, 1)
#         y_var = np.array(y_var).reshape(-1, 1)

#         # if the data is not empty, perform machine learning
#         if len(x) > 0:

#             # initialize the machine learning models for
#             train_x = torch.tensor(x).reshape(-1, 4)
#             train_y = torch.tensor(np.concatenate([y_mean], axis=1))
#             train_y_var = torch.tensor(np.concatenate([y_var], axis=1))

#             # print(train_x)
#             # print(train_y)
#             # print(train_y_var)

#             # initialize the model
#             mll, model = botorch_tool.tools.initialize_heter_noise_model(
#                 train_x=train_x, train_obj=train_y, train_obj_var=train_y_var
#             )
#             # fit the model
#             try:
#                 fit_gpytorch_mll(mll)
#             except:
#                 print(f"fitting noiseless model for combination {current_combo}")
#                 try:
#                     # initialize the noiseless model (not really noiseless, because the noise is estimated by the means)
#                     mll, model = botorch_tool.tools.initialize_noiseless_model(
#                         train_x=train_x, train_obj=train_y
#                     )
#                     fit_gpytorch_mll(mll)
#                 except:
#                     raise ValueError(
#                         f"model fitting failed on combination {current_combo}!"
#                     )

#             # get the bounds for the optimization
#             bounds = torch.stack([torch.zeros(4), torch.ones(4)])
#             # sample the space according to the fitted model to perform BO
#             sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
#             acq_func = qNoisyExpectedImprovement(model, train_x, sampler)
#             # optimize the acquisition function to increase the y value
#             candidate, expected_y = optimize_acqf(
#                 acq_function=acq_func,
#                 bounds=bounds,
#                 q=1,
#                 num_restarts=200,
#                 raw_samples=1024,
#                 inequality_constraints=[
#                     (torch.tensor([0, 1]), torch.tensor([-1.0, -1.0]), -1.0)
#                 ],
#             )
#             print(f"{candidate} {expected_y}")

#             candidate = candidate.detach().cpu().numpy()[0]
#             # get a new row for the candidate generation
#             candidate_row = exp_data.iloc[exp_index].copy()
            
#             volume1 = candidate[0] * 9.5 + 0.5
#             volume2 = candidate[1] * 9.5 + 0.5
#             volume3 = candidate[2] * 5
#             volume4 = candidate[3] * 5
#             volume5 = 20 - volume1 - volume2 - volume3 - volume4
            
#             volume1 = np.around(volume1, 2)
#             volume2 = np.around(volume2, 2)
#             volume3 = np.around(volume3, 2)
#             volume4 = np.around(volume4, 2)
#             volume5 = np.around(volume5, 2)
            
#             candidate_row["volume1"] = volume1
#             candidate_row["volume2"] = volume2
#             candidate_row["volume3"] = volume3
#             candidate_row["volume4"] = volume4
#             candidate_row["volume5"] = volume5
    
#             candidate_row["syn_completed"] = "None"
#             candidate_row["storage_location"] = "None"
#             candidate_row["in_storage"] = "None"
#             candidate_row["cat_completed"] = "None"
#             candidate_row["lsv_overpotential_-0.01"] = "None"
#             candidate_row["lsv_overpotential_-0.02"] = "None"
#             candidate_row["lsv_overpotential_-0.03"] = "None"
#             candidate_row["lsv_overpotential_-0.04"] = "None"
#             candidate_row["lsv_overpotential_-0.05"] = "None"

#             candidate_row = candidate_row.replace({np.nan: None, "None": None})

#             return candidate_row, repeat_exp

#         else:
#             return "not_ready", None
#     else:
#         return "not_ready", None
    
def generate_new_exp_SO_one_system(exp_data: pd.DataFrame, exp_index: int):
    """
    Generate a series of new experiments from the previous data and the exp_index for a single objective optimization
    Args:
        exp_data: a pandas.dataframe that records the experiments
        exp_index: the index of the experiments when we call generate_new_exp, which determines the categories of the reactants
    Returns:
        the next sampling points or None
    """
    # set up the random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # get the current combo of the experiments
    current_combo = exp_data[["substance1", "substance2"]].iloc[exp_index]
    identical_index = (exp_data[["substance1", "substance2"]] == current_combo).all(
        axis=1
    )
    selected_index = np.where(identical_index == True)[0]

    # check if all the experiemnts are characterized for this specific combo
    if sum(exp_data.loc[identical_index]["lsv_overpotential_-0.02"].values != None) >=8:
        # read in the data
        x = []
        y_mean = []
        y_var = []
        # get the selected dataset
        for j in selected_index:
            # do nothing for this point as it is not characterized, or it is a reference data point
            if (exp_data.iloc[j]["lsv_overpotential_-0.02"] == None) or ((repeat_exp[0:10] == exp_data.iloc[j][0:10]).all()):
                pass 
            else:
                # append the x value first
                x_temp = [
                    (exp_data.iloc[j]["volume1"] - 0.5) / 9.5,
                    (exp_data.iloc[j]["volume2"] - 0.5) / 9.5,
                    (exp_data.iloc[j]["volume3"] / 5),
                    (exp_data.iloc[j]["volume4"] / 5),
                ]
                x.append(x_temp)
                # We assign a normal value to it if the results are not "unknown"
                if exp_data.iloc[j]["lsv_overpotential_-0.02"] != "unknown":
                    y_mean.append(1 * exp_data.iloc[j]["lsv_overpotential_-0.02"])
                    
                # if the results are unknown, it is a bad point and we should assign a very low value to it
                else:
                    y_mean.append(-1.30)
                # append a 30mV variance to the data (corresponding to a std of 10 mV)
                y_var.append((0.03**2))

        # process the data input numpy array
        x = np.array(x).reshape(-1, 4)
        y_mean = np.array(y_mean).reshape(-1, 1)
        y_var = np.array(y_var).reshape(-1, 1)

        # if the data is not empty, perform machine learning
        if len(x) > 0:

            # initialize the machine learning models for
            train_x = torch.tensor(x).reshape(-1, 4)
            train_y = torch.tensor(np.concatenate([y_mean], axis=1))
            train_y_var = torch.tensor(np.concatenate([y_var], axis=1))

            # print(train_x)
            # print(train_y)
            # print(train_y_var)

            # initialize the model
            mll, model = botorch_tool.tools.initialize_heter_noise_model(
                train_x=train_x, train_obj=train_y, train_obj_var=train_y_var
            )
            # fit the model
            try:
                fit_gpytorch_mll(mll)
            except:
                print(f"fitting noiseless model for combination {current_combo}")
                try:
                    # initialize the noiseless model (not really noiseless, because the noise is estimated by the means)
                    mll, model = botorch_tool.tools.initialize_noiseless_model(
                        train_x=train_x, train_obj=train_y
                    )
                    fit_gpytorch_mll(mll)
                except:
                    raise ValueError(
                        f"model fitting failed on combination {current_combo}!"
                    )

            # get the bounds for the optimization
            bounds = torch.stack([torch.zeros(4), torch.ones(4)])
            
            # sample the space according to the fitted model to perform BO
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
            acq_func = qLogNoisyExpectedImprovement(model, train_x, sampler)
            
            # optimize the acquisition function to increase the y value
            candidate, expected_y = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=3,
                num_restarts=20,
                raw_samples=1024,
                inequality_constraints=[
                    (torch.tensor([0, 1]), torch.tensor([-1.0, -1.0]), -1.0)
                ],
            )
            # we should get three optimized samples 
            print(f"{candidate} {expected_y}")

            # change the candidates to numpy format 
            candidate_total = candidate.detach().cpu().numpy()
            
            # create a list to store pandas.series
            candidate_rows_total = [] 
            for candidate in candidate_total:         
                # get a new row for the candidate generation
                candidate_row = exp_data.iloc[exp_index].copy()
                
                volume1 = candidate[0] * 9.5 + 0.5
                volume2 = candidate[1] * 9.5 + 0.5
                volume3 = candidate[2] * 5
                volume4 = candidate[3] * 5
                volume5 = 20 - volume1 - volume2 - volume3 - volume4
                
                volume1 = np.around(volume1, 1)
                volume2 = np.around(volume2, 1)
                volume3 = np.around(volume3, 1)
                volume4 = np.around(volume4, 1)
                volume5 = np.around(volume5, 1)
                
                candidate_row["volume1"] = volume1
                candidate_row["volume2"] = volume2
                candidate_row["volume3"] = volume3
                candidate_row["volume4"] = volume4
                candidate_row["volume5"] = volume5
        
                candidate_row["syn_completed"] = "None"
                candidate_row["storage_location"] = "None"
                candidate_row["in_storage"] = "None"
                candidate_row["cat_completed"] = "None"
                candidate_row["lsv_overpotential_-0.01"] = "None"
                candidate_row["lsv_overpotential_-0.02"] = "None"
                candidate_row["lsv_overpotential_-0.03"] = "None"
                candidate_row["lsv_overpotential_-0.04"] = "None"
                candidate_row["lsv_overpotential_-0.05"] = "None"
                # update the row 
                candidate_row = candidate_row.replace({np.nan: None, "None": None})
                # append it to list 
                candidate_rows_total.append(candidate_row)
                
            return candidate_rows_total, repeat_exp

        else:
            return "not_ready", None
    else:
        return "not_ready", None
    
    
def generate_new_exp_MO_one_system(exp_data: pd.DataFrame, exp_index: int):
    
    """
    Generate a series of new experiments from the previous data and the exp_index for multi-objective optimization
    Args:
        exp_data: a pandas.dataframe that records the experiments
        exp_index: the index of the experiments when we call generate_new_exp, which determines the categories of the reactants
    Returns:
        the next sampling points or None
    """
    # set up the random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # get the current combo of the experiments
    current_combo = exp_data[["substance1", "substance2"]].iloc[exp_index]
    identical_index = (exp_data[["substance1", "substance2"]] == current_combo).all(
        axis=1
    )
    selected_index = np.where(identical_index == True)[0]
    
    # set up the reference experiment
    repeat_exp = pd.Series(
        {
            "substance1": current_combo["substance1"],
            "volume1": 20*6/8,
            "substance2": current_combo["substance2"],
            "volume2": 20*1/8,
            "substance3": "Na3Cit",
            "volume3": 20*1/8,
            
            
            "syn_completed": "None",
            "storage_location": "None",
            "in_storage": "None",
            "cat_completed": "None",
            "lsv_overpotential_-0.01": "None",
            "lsv_overpotential_-0.02": "None",
            "lsv_overpotential_-0.03": "None",
            "lsv_overpotential_-0.04": "None",
            "lsv_overpotential_-0.05": "None",
            "ir_value": "None",
            
            "mass_time_start": "None",
            "mass_time_end": "None",
            "separation_coefficient":"None"
        }
    )
    repeat_exp = repeat_exp.replace({np.nan: None, "None": None})
    
    # check if all the experiemnts are characterized for this specific combo
    if sum(exp_data.loc[identical_index]["lsv_overpotential_-0.01"].values != None) >=8:
        # read in the data
        x = []
        # store the current data
        y_mean = []
        y_var = []
        
        # store the seperation coefficient data 
        y_sc_mean = []
        y_sc_var = []
        # get the selected dataset
        for j in selected_index:
            # do nothing for this point as it is not characterized, or it is a reference data point
            if (exp_data.iloc[j]["lsv_overpotential_-0.01"] == None) or ((repeat_exp[0:6] == exp_data.iloc[j][0:6]).all()):
                pass 
            else:
                # append the x value first
                x_temp = [
                    (exp_data.iloc[j]["volume1"] - 1) / 18,
                    (exp_data.iloc[j]["volume2"] - 1) / 18,
                ]
                x.append(x_temp)
                
                # deal with the current data
                # We assign a normal value to it if the results are not "unknown"
                if exp_data.iloc[j]["lsv_overpotential_-0.01"] != "unknown":
                    y_mean.append(1 * exp_data.iloc[j]["lsv_overpotential_-0.01"])
                # if the results are unknown, it is a bad point and we should assign a very low value to it
                else:
                    y_mean.append(-1.30)
                # append a 30 mV variance to the data (corresponding to a std of 10 mV)
                y_var.append((0.03**2))
                
                # deal with the separation coefficient data
                y_sc_mean.append(1 * exp_data.iloc[j]["separation_coefficient"])

                # append a 0.2 variance to the data
                y_sc_var.append((0.2**2))
                

        # process the data input numpy array
        x = np.array(x).reshape(-1, 2)
        y_mean = np.array(y_mean).reshape(-1, 1)
        y_var = np.array(y_var).reshape(-1, 1)
        y_sc_mean = np.array(y_sc_mean).reshape(-1,1)
        y_sc_var = np.array(y_sc_var).reshape(-1,1)

        # if the data is not empty, perform machine learning
        if len(x) > 0:

            # initialize the machine learning models for
            train_x = torch.tensor(x).reshape(-1, 2)
                        
            train_y = torch.tensor(np.concatenate([y_mean + 1.3, y_sc_mean], axis = 1))
            
            train_y_var = torch.tensor(np.concatenate([y_var,y_sc_var], axis = 1))

            # initialize the model
            mll, model = botorch_tool.tools.initialize_heter_noise_model(
                train_x=train_x, train_obj=train_y, train_obj_var=train_y_var
            )
            # fit the model
            try:
                fit_gpytorch_mll(mll)
            except:
                print(f"fitting noiseless model for combination {current_combo}")
                try:
                    # initialize the noiseless model (not really noiseless, because the noise is estimated by the means)
                    mll, model = botorch_tool.tools.initialize_noiseless_model(
                        train_x=train_x, train_obj=train_y
                    )
                    fit_gpytorch_mll(mll)
                except:
                    raise ValueError(
                        f"model fitting failed on combination {current_combo}!"
                    )

            # get the bounds for the optimization
            bounds = torch.stack([torch.zeros(2), torch.ones(2)])
        
            # sample the space according to the fitted model to perform BO
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
                        
            # partition non-dominated space into disjoint rectangles
            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=[0, 0],  # use known reference point
                X_baseline=train_x,
                prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
                sampler=sampler,
            )
            # optimize the acquisition function to increase the y value
            candidate, expected_y = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=3,
                num_restarts=20,
                raw_samples=1024,
                inequality_constraints=[
                    (torch.tensor([0, 1]), torch.tensor([-1.0, -1.0]), -1.0)
                ],
            )
            
            # we should get three optimized samples 
            print(f"{candidate} {expected_y}")

            # change the candidates to numpy format 
            candidate_total = candidate.detach().cpu().numpy()
            
            # create a list to store pandas.series
            candidate_rows_total = [] 
            for candidate in candidate_total:         
                # get a new row for the candidate generation
                candidate_row = exp_data.iloc[exp_index].copy()
                
                volume1 = candidate[0] * 18.0 + 1.0
                volume2 = candidate[1] * 18.0 + 1.0
                volume3 = 20 - volume1 - volume2

                
                volume1 = np.around(volume1, 1)
                volume2 = np.around(volume2, 1)
                volume3 = np.around(volume3, 1)
                
                candidate_row["volume1"] = volume1
                candidate_row["volume2"] = volume2
                candidate_row["volume3"] = volume3
        
                candidate_row["syn_completed"] = "None"
                candidate_row["storage_location"] = "None"
                candidate_row["in_storage"] = "None"
                candidate_row["cat_completed"] = "None"
                candidate_row["lsv_overpotential_-0.01"] = "None"
                candidate_row["lsv_overpotential_-0.02"] = "None"
                candidate_row["lsv_overpotential_-0.03"] = "None"
                candidate_row["lsv_overpotential_-0.04"] = "None"
                candidate_row["lsv_overpotential_-0.05"] = "None"
                
                
                candidate_row["mass_time_start"] = "None"
                candidate_row["mass_time_end"] = "None"
                candidate_row["separation_coefficient"] = "None" 
                         
                # update the row 
                candidate_row = candidate_row.replace({np.nan: None, "None": None})
                
                # append it to list 
                candidate_rows_total.append(candidate_row)
                
            return candidate_rows_total, repeat_exp

        else:
            return "not_ready", None
    else:
        return "not_ready", None
    
def generate_new_exp_MO_one_system_V2(exp_data: pd.DataFrame, exp_index: int):
    
    """
    Generate a series of new experiments from the previous data and the exp_index for multi-objective optimization of bimetallic catalysts
    Args:
        exp_data: a pandas.dataframe that records the experiments
        exp_index: the index of the experiments when we call generate_new_exp, which determines the categories of the reactants
    Returns:
        the next sampling points or None
    """
    # set up the random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # get the current combo of the experiments
    current_combo = exp_data[["substance1", "substance2"]].iloc[exp_index]
    identical_index = (exp_data[["substance1", "substance2"]] == current_combo).all(
        axis=1
    )
    selected_index = np.where(identical_index == True)[0]
    
    # set up the reference experiment
    repeat_exp = pd.Series(
        {
            "substance1": current_combo["substance1"],
            "volume1": 20*6/8,
            "substance2": current_combo["substance2"],
            "volume2": 20*1/8,
            "substance3": "Na3Cit",
            "volume3": 20*1/8,
            "substance4": "water",
            "volume4": 0.0,
            
            "syn_completed": "None",
            "storage_location": "None",
            "in_storage": "None",
            "cat_completed": "None",
            "lsv_overpotential_-0.01": "None",
            "lsv_overpotential_-0.02": "None",
            "lsv_overpotential_-0.03": "None",
            "lsv_overpotential_-0.04": "None",
            "lsv_overpotential_-0.05": "None",
            "ir_value": "None",
            
            "mass_time_start": "None",
            "mass_time_end": "None",
            "separation_coefficient":"None"
        }
    )
    repeat_exp = repeat_exp.replace({np.nan: None, "None": None})
    
    # check if all the experiemnts are characterized for this specific combo
    if sum(exp_data.loc[identical_index]["lsv_overpotential_-0.01"].values != None) >= 8:
        # read in the data
        x = []
        # store the current data
        y_mean = []
        y_var = []
        
        # store the seperation coefficient data 
        y_sc_mean = []
        y_sc_var = []
        # get the selected dataset
        for j in selected_index:
            # do nothing for this point as it is not characterized, or it is a reference data point
            if (exp_data.iloc[j]["lsv_overpotential_-0.01"] == None) or ((repeat_exp[0:8] == exp_data.iloc[j][0:8]).all()):
                pass 
            else:
                # append the x value first
                x_temp = [
                    (exp_data.iloc[j]["volume1"] - 1) / 18,
                    (exp_data.iloc[j]["volume2"] - 1) / 18,
                    (exp_data.iloc[j]["volume3"]) / 18,
                ]
                
                x.append(x_temp)
                
                # deal with the current data
                # we assign a normal value to it if the results are not "unknown"
                if exp_data.iloc[j]["lsv_overpotential_-0.01"] != "unknown":
                    y_mean.append(1 * exp_data.iloc[j]["lsv_overpotential_-0.01"])
                # if the results are unknown, it is a bad point and we should assign a very low value to it
                else:
                    y_mean.append(-1.20)
                # append a 30 mV variance to the data (corresponding to a std of 10 mV)
                y_var.append((0.03**2))
                
                # deal with the separation coefficient data
                y_sc_mean.append(1 * exp_data.iloc[j]["separation_coefficient"])

                # append a 0.2 variance to the data
                y_sc_var.append((0.2**2))
                

        # process the data input numpy array
        x = np.array(x).reshape(-1, 3)
        y_mean = np.array(y_mean).reshape(-1, 1)
        y_var = np.array(y_var).reshape(-1, 1)
        y_sc_mean = np.array(y_sc_mean).reshape(-1,1)
        y_sc_var = np.array(y_sc_var).reshape(-1,1)

        # if the data is not empty, perform machine learning
        if len(x) > 0:

            # initialize the machine learning models for
            train_x = torch.tensor(x).reshape(-1, 3)
                        
            train_y = torch.tensor(np.concatenate([y_mean + 1.2, y_sc_mean], axis = 1))
            
            train_y_var = torch.tensor(np.concatenate([y_var,y_sc_var], axis = 1))

            # initialize the model
            mll, model = botorch_tool.tools.initialize_heter_noise_model(
                train_x=train_x, train_obj=train_y, train_obj_var=train_y_var
            )
            # fit the model
            try:
                fit_gpytorch_mll(mll)
            except:
                print(f"fitting noiseless model for combination {current_combo}")
                try:
                    # initialize the noiseless model (not really noiseless, because the noise is estimated by the means)
                    mll, model = botorch_tool.tools.initialize_noiseless_model(
                        train_x=train_x, train_obj=train_y
                    )
                    fit_gpytorch_mll(mll)
                except:
                    raise ValueError(
                        f"model fitting failed on combination {current_combo}!"
                    )

            # get the bounds for the optimization
            bounds = torch.stack([torch.zeros(3), torch.ones(3)])
        
            # sample the space according to the fitted model to perform BO
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
                        
            # partition non-dominated space into disjoint rectangles
            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=[0, 0],  # use known reference point
                X_baseline=train_x,
                prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
                sampler=sampler,
            )
            # optimize the acquisition function to increase the y value
            candidate, expected_y = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=3,
                num_restarts=20,
                raw_samples=1024,
                inequality_constraints=[
                    (torch.tensor([0, 1, 2]), torch.tensor([-1.0, -1.0, -1.0]), -1.0)
                ],
            )
            
            # we should get three optimized samples 
            print(f"{candidate} {expected_y}")

            # change the candidates to numpy format 
            candidate_total = candidate.detach().cpu().numpy()
            
            # create a list to store pandas.series
            candidate_rows_total = [] 
            for candidate in candidate_total:         
                # get a new row for the candidate generation
                candidate_row = exp_data.iloc[exp_index].copy()
                
                volume1 = candidate[0] * 18.0 + 1.0
                volume2 = candidate[1] * 18.0 + 1.0
                volume3 = candidate[2] * 18.0
                volume4 = 20 - volume1 - volume2 - volume3 

                
                volume1 = np.around(volume1, 1)
                volume2 = np.around(volume2, 1)
                volume3 = np.around(volume3, 1)
                volume4 = np.around(volume4, 1)

                
                candidate_row["volume1"] = volume1
                candidate_row["volume2"] = volume2
                candidate_row["volume3"] = volume3
                candidate_row["volume4"] = volume4

                candidate_row["syn_completed"] = "None"
                candidate_row["storage_location"] = "None"
                candidate_row["in_storage"] = "None"
                candidate_row["cat_completed"] = "None"
                candidate_row["lsv_overpotential_-0.01"] = "None"
                candidate_row["lsv_overpotential_-0.02"] = "None"
                candidate_row["lsv_overpotential_-0.03"] = "None"
                candidate_row["lsv_overpotential_-0.04"] = "None"
                candidate_row["lsv_overpotential_-0.05"] = "None"
                candidate_row["ir_value"] = "None"
                
                candidate_row["mass_time_start"] = "None"
                candidate_row["mass_time_end"] = "None"
                candidate_row["separation_coefficient"] = "None" 
                         
                # update the row 
                candidate_row = candidate_row.replace({np.nan: None, "None": None})
                
                # append it to list 
                candidate_rows_total.append(candidate_row)
            
            return candidate_rows_total, repeat_exp

        else:
            return "not_ready", None
    else:
        return "not_ready", None
    
def generate_new_exp_SO_one_system_V3(exp_data: pd.DataFrame, exp_index: int):
    
    """
    Generate a series of new experiments from the previous data and the exp_index for single-objective optimization
    Args:
        exp_data: a pandas.dataframe that records the experiments
        exp_index: the index of the experiments when we call generate_new_exp, which determines the categories of the reactants
    Returns:
        the next sampling points or None
    """
    # set up the random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # get the current combo of the experiments
    current_combo = exp_data[["substance1", "substance2"]].iloc[exp_index]
    identical_index = (exp_data[["substance1", "substance2"]] == current_combo).all(
        axis=1
    )
    selected_index = np.where(identical_index == True)[0]
    
    # set up the reference experiment
    repeat_exp = pd.Series(
        {
            "substance1": current_combo["substance1"],
            "volume1": 8*6/8,
            "substance2": current_combo["substance2"],
            "volume2": 8*1/8,
            "substance3": "Na3Cit",
            "volume3": 8*1/8,
            "substance4": "water",
            "volume4": 0.0,
            
            "syn_completed": "None",
            "storage_location": "None",
            "in_storage": "None",
            "cat_completed": "None",
            "lsv_overpotential_-0.01": "None",
            "lsv_overpotential_-0.02": "None",
            "lsv_overpotential_-0.03": "None",
            "lsv_overpotential_-0.04": "None",
            "lsv_overpotential_-0.05": "None",
            "ir_value": "None",
            
            "mass_time_start": "None",
            "mass_time_end": "None",
            "separation_coefficient":"None"
        }
    )
    repeat_exp = repeat_exp.replace({np.nan: None, "None": None})
    
    # check if all the experiemnts are characterized for this specific combo
    if sum(exp_data.loc[identical_index]["separation_coefficient"].values != None) >= 8:
        # read in the data
        x = []

        # store the seperation coefficient data 
        y_sc_mean = []
        y_sc_var = []
        # get the selected dataset
        for j in selected_index:
            # do nothing for this point as it is not characterized, or it is a reference data point
            if (exp_data.iloc[j]["separation_coefficient"] == None) or ((repeat_exp[0:8] == exp_data.iloc[j][0:8]).all()):
                pass 
            else:
                # append the x value first
                x_temp = [
                    (exp_data.iloc[j]["volume1"] - 1) / 18,
                    (exp_data.iloc[j]["volume2"] - 1) / 18,
                    (exp_data.iloc[j]["volume3"]) / 18,
                ]
                
                x.append(x_temp)
                
                # deal with the current data
                
                # deal with the separation coefficient data
                y_sc_mean.append(1 * exp_data.iloc[j]["separation_coefficient"])

                # append a 0.2 variance to the data
                y_sc_var.append((0.2**2))
                

        # process the data input numpy array
        x = np.array(x).reshape(-1, 3)
        y_sc_mean = np.array(y_sc_mean).reshape(-1,1)
        y_sc_var = np.array(y_sc_var).reshape(-1,1)

        # if the data is not empty, perform machine learning
        if len(x) > 0:

            # initialize the machine learning models for
            train_x = torch.tensor(x).reshape(-1, 3)
                        
            train_y = torch.tensor(np.concatenate([15-y_sc_mean], axis = 1))
            
            train_y_var = torch.tensor(np.concatenate([y_sc_var], axis = 1))

            # initialize the model
            mll, model = botorch_tool.tools.initialize_heter_noise_model(
                train_x=train_x, train_obj=train_y, train_obj_var=train_y_var
            )
            # fit the model
            try:
                fit_gpytorch_mll(mll)
            except:
                print(f"fitting noiseless model for combination {current_combo}")
                try:
                    # initialize the noiseless model (not really noiseless, because the noise is estimated by the means)
                    mll, model = botorch_tool.tools.initialize_noiseless_model(
                        train_x=train_x, train_obj=train_y
                    )
                    fit_gpytorch_mll(mll)
                except:
                    raise ValueError(
                        f"model fitting failed on combination {current_combo}!"
                    )

            # get the bounds for the optimization
            bounds = torch.stack([torch.zeros(3), torch.ones(3)])
        
            # sample the space according to the fitted model to perform BO
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
            acq_func = qLogNoisyExpectedImprovement(model, train_x, sampler)
            
            # optimize the acquisition function to increase the y value
            candidate, expected_y = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=3,
                num_restarts=20,
                raw_samples=1024,
                inequality_constraints=[
                    (torch.tensor([0, 1, 2]), torch.tensor([-1.0, -1.0, -1.0]), -1.0)
                ],
            )
            
            # we should get three optimized samples 
            print(f"{candidate} {expected_y}")

            # change the candidates to numpy format 
            candidate_total = candidate.detach().cpu().numpy()
            
            # create a list to store pandas.series
            candidate_rows_total = [] 
            for candidate in candidate_total:         
                # get a new row for the candidate generation
                candidate_row = exp_data.iloc[exp_index].copy()
                
                volume1 = candidate[0] * 18.0 + 1.0
                volume2 = candidate[1] * 18.0 + 1.0
                volume3 = candidate[2] * 18.0
                volume4 = 20 - volume1 - volume2 - volume3 

                volume1 = (8/20)*volume1
                volume2 = (8/20)*volume2
                volume3 = (8/20)*volume3
                volume4 = (8/20)*volume4

                
                volume1 = np.around(volume1, 1)
                volume2 = np.around(volume2, 1)
                volume3 = np.around(volume3, 1)
                volume4 = np.around(volume4, 1)

                
                candidate_row["volume1"] = volume1
                candidate_row["volume2"] = volume2
                candidate_row["volume3"] = volume3
                candidate_row["volume4"] = volume4

                candidate_row["syn_completed"] = "None"
                candidate_row["storage_location"] = "None"
                candidate_row["in_storage"] = "None"
                candidate_row["cat_completed"] = "None"
                candidate_row["lsv_overpotential_-0.01"] = "None"
                candidate_row["lsv_overpotential_-0.02"] = "None"
                candidate_row["lsv_overpotential_-0.03"] = "None"
                candidate_row["lsv_overpotential_-0.04"] = "None"
                candidate_row["lsv_overpotential_-0.05"] = "None"
                candidate_row["ir_value"] = "None"
                
                candidate_row["mass_time_start"] = "None"
                candidate_row["mass_time_end"] = "None"
                candidate_row["separation_coefficient"] = "None" 
                         
                # update the row 
                candidate_row = candidate_row.replace({np.nan: None, "None": None})
                
                # append it to list 
                candidate_rows_total.append(candidate_row)
            
            return candidate_rows_total, repeat_exp

        else:
            return "not_ready", None
    else:
        return "not_ready", None

def generate_new_exp_MO_one_system_V4(exp_data: pd.DataFrame, exp_index: int):
    
    """
    Generate a series of new experiments for full water splitting.
    """
    # set up the random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # get the current combo of the experiments
    current_combo = exp_data[["substance1", "substance2"]].iloc[exp_index]
    identical_index = (exp_data[["substance1", "substance2"]] == current_combo).all(
        axis=1
    )
    selected_index = np.where(identical_index == True)[0]
    
    # set up the reference experiment
    repeat_exp = pd.Series(
        {
            "substance1": current_combo["substance1"],
            "volume1": 20*2/10,
            "substance2": current_combo["substance2"],
            "volume2": 20*2/10,
            "substance3": "Na3Cit",
            "volume3": 20*2/10,
            "substance4": "NaH2PO2",
            "volume4": 20*2/10,
            "substance5": "water",
            "volume5": 20*2/10,

            "syn_completed": "None",
            "storage_location": "None",
            "in_storage": "None",
            "cat_completed": "None",
            
            "ir_value":"None",
            "lsv_overpotential_-0.01": "None",
            "lsv_overpotential_-0.02": "None",
            "lsv_overpotential_-0.03": "None",
            "lsv_overpotential_-0.04": "None",
            "lsv_overpotential_-0.05": "None",

            "lsv_overpotential_0.01": "None",
            "lsv_overpotential_0.02": "None",
            "lsv_overpotential_0.03": "None",
            "lsv_overpotential_0.04": "None",
            "lsv_overpotential_0.05": "None",
            

            "mass_time_start": "None",
            "mass_time_end": "None",
            "separation_coefficient":"None"
        }
    )
    repeat_exp = repeat_exp.replace({np.nan: None, "None": None})
    
    # check if all the experiemnts are characterized for this specific combo
    if sum(exp_data.loc[identical_index]["lsv_overpotential_-0.01"].values != None) >=8:
        # read in the data
        x = []
        # store the current data
        y_mean = []
        y_var = []
        
        # store the seperation coefficient data 
        y_sc_mean = []
        y_sc_var = []
        # get the selected dataset
        for j in selected_index:
            # do nothing for this point as it is not characterized, or it is a reference data point
            if (exp_data.iloc[j]["lsv_overpotential_-0.01"] == None) or ((repeat_exp[0:10] == exp_data.iloc[j][0:10]).all()):
                pass 
            else:
                # append the x value first
                x_temp = [
                    (exp_data.iloc[j]["volume1"] - 1) / 18,
                    (exp_data.iloc[j]["volume2"] - 1) / 18,
                    (exp_data.iloc[j]["volume3"]) / 18,
                    (exp_data.iloc[j]["volume4"]) / 18,
                ]
                
                x.append(x_temp)
                
                # deal with the current data
                # we assign a normal value to it if the results are not "unknown"
                if exp_data.iloc[j]["lsv_overpotential_-0.01"] != "unknown":
                    y_mean.append(1 * exp_data.iloc[j]["lsv_overpotential_-0.01"])
                # if the results are unknown, it is a bad point and we should assign a very low value to it
                else:
                    y_mean.append(-1.20)
                # append a 30 mV variance to the data (corresponding to a std of 10 mV)
                y_var.append((0.03**2))

                if exp_data.iloc[j]["lsv_overpotential_0.01"] != "unknown":
                    y_sc_mean.append(-1 * exp_data.iloc[j]["lsv_overpotential_0.01"]) # notice the negative sign
                # if the results are unknown, it is a bad point and we should assign a very low value to it
                else:
                    y_sc_mean.append(-1 * 0.80)

                # append a 30 mV variance to the data (corresponding to a std of 10 mV)
                y_sc_var.append((0.03**2))
                

        # process the data input numpy array
        x = np.array(x).reshape(-1, 4)
        y_mean = np.array(y_mean).reshape(-1, 1)
        y_var = np.array(y_var).reshape(-1, 1)
        y_sc_mean = np.array(y_sc_mean).reshape(-1,1)
        y_sc_var = np.array(y_sc_var).reshape(-1,1)

        # if the data is not empty, perform machine learning
        if len(x) > 0:

            # initialize the machine learning models for
            train_x = torch.tensor(x).reshape(-1, 4)
                        
            train_y = torch.tensor(np.concatenate([y_mean + 1.3, y_sc_mean + 0.8], axis = 1)) # so that the minimum is 0 
            
            train_y_var = torch.tensor(np.concatenate([y_var,y_sc_var], axis = 1))

            # initialize the model
            mll, model = botorch_tool.tools.initialize_heter_noise_model(
                train_x=train_x, train_obj=train_y, train_obj_var=train_y_var
            )
            # fit the model
            try:
                fit_gpytorch_mll(mll)
            except:
                print(f"fitting noiseless model for combination {current_combo}")
                try:
                    # initialize the noiseless model (not really noiseless, because the noise is estimated by the means)
                    mll, model = botorch_tool.tools.initialize_noiseless_model(
                        train_x=train_x, train_obj=train_y
                    )
                    fit_gpytorch_mll(mll)
                except:
                    raise ValueError(
                        f"model fitting failed on combination {current_combo}!"
                    )

            # get the bounds for the optimization
            bounds = torch.stack([torch.zeros(4), torch.ones(4)])
        
            # sample the space according to the fitted model to perform BO
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
                        
            # partition non-dominated space into disjoint rectangles
            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=[0, 0],  # use known reference point
                X_baseline=train_x,
                prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
                sampler=sampler,
            )
            # optimize the acquisition function to increase the y value
            candidate, expected_y = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=3,
                num_restarts=20,
                raw_samples=1024,
                inequality_constraints=[
                    (torch.tensor([0, 1, 2, 3]), torch.tensor([-1.0, -1.0, -1.0, -1.0]), -1.0)
                ],
            )
            
            # we should get three optimized samples 
            print(f"{candidate} {expected_y}")

            # change the candidates to numpy format 
            candidate_total = candidate.detach().cpu().numpy()
            
            # create a list to store pandas.series
            candidate_rows_total = [] 
            for candidate in candidate_total:         
                # get a new row for the candidate generation
                candidate_row = exp_data.iloc[exp_index].copy()
                
                volume1 = candidate[0] * 18.0 + 1.0
                volume2 = candidate[1] * 18.0 + 1.0
                volume3 = candidate[2] * 18.0
                volume4 = candidate[3] * 18.0
                volume5 = 20 - volume1 - volume2 - volume3 - volume4

                
                volume1 = np.around(volume1, 1)
                volume2 = np.around(volume2, 1)
                volume3 = np.around(volume3, 1)
                volume4 = np.around(volume4, 1)
                volume5 = np.around(volume5, 1)

                
                candidate_row["volume1"] = volume1
                candidate_row["volume2"] = volume2
                candidate_row["volume3"] = volume3
                candidate_row["volume4"] = volume4
                candidate_row["volume5"] = volume5

                candidate_row["syn_completed"] = "None"
                candidate_row["storage_location"] = "None"
                candidate_row["in_storage"] = "None"
                candidate_row["cat_completed"] = "None"
                candidate_row["lsv_overpotential_-0.01"] = "None"
                candidate_row["lsv_overpotential_-0.02"] = "None"
                candidate_row["lsv_overpotential_-0.03"] = "None"
                candidate_row["lsv_overpotential_-0.04"] = "None"
                candidate_row["lsv_overpotential_-0.05"] = "None"

                candidate_row["lsv_overpotential_0.01"] = "None"
                candidate_row["lsv_overpotential_0.02"] = "None"
                candidate_row["lsv_overpotential_0.03"] = "None"
                candidate_row["lsv_overpotential_0.04"] = "None"
                candidate_row["lsv_overpotential_0.05"] = "None"
                candidate_row["ir_value"] = "None"

                candidate_row["mass_time_start"] = "None"
                candidate_row["mass_time_end"] = "None"
                candidate_row["separation_coefficient"] = "None" 
                         
                # update the row 
                candidate_row = candidate_row.replace({np.nan: None, "None": None})
                
                # append it to list 
                candidate_rows_total.append(candidate_row)
            
            return candidate_rows_total, repeat_exp

        else:
            return "not_ready", None
    else:
        return "not_ready", None
    
import numpy as np
import pandas as pd
import torch

def generate_new_exp_P(exp_data: pd.DataFrame, exp_index: int):
    """
    Generate a series of new experiments from the previous data and the exp_index for multi-objective optimization of metal phosphide catalysts.
    Args:
        exp_data: a pandas DataFrame that records the experiments.
        exp_index: the index of the experiments when we call generate_new_exp, which determines the categories of the reactants.
    Returns:
        The next sampling points or None.
    """
    # Set up the random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Get the current combination of the experiments
    current_combo = exp_data[["substance1", "substance2"]].iloc[exp_index]
    identical_index = (exp_data[["substance1", "substance2"]] == current_combo).all(axis=1)
    selected_index = np.where(identical_index == True)[0]
    
    # Set up the reference experiment  
    repeat_exp = pd.Series(
        {
            "substance1": current_combo["substance1"],
            "volume1": float(np.around(6 * 9 / 11, 1)),
            "substance2": current_combo["substance2"],
            "volume2": float(np.around(2 * 9 / 11, 1)),
            "substance3": "Na3Cit",
            "volume3": float(np.around(2 * 9 / 11, 1)),
            "substance4": "NaH2PO2",
            "volume4": float(np.around(1 * 9 / 11, 1)),
            "substance5": "water",
            "volume5": float(np.around(0.0, 1)),
            "current": int(-300),
            "syn_completed": "None",
            "storage_location": "None",
            "in_storage": "None",
            "cat_completed": "None",
            "lsv_overpotential_-0.01": "None",
            "lsv_overpotential_-0.02": "None",
            "lsv_overpotential_-0.03": "None",
            "lsv_overpotential_-0.04": "None",
            "lsv_overpotential_-0.05": "None",
            "ir_value": "None",
            "mass_time_start": "None",
            "mass_time_end": "None",
            "separation_coefficient": "None"
        }
    )
    repeat_exp = repeat_exp.replace({np.nan: None, "None": None})
    
    # Check if all the experiments are characterized for this specific combo
    if sum(exp_data.loc[identical_index]["lsv_overpotential_-0.01"].values != None) >= 8:
        # Read in the data
        x = []

        # Store the current data
        y_mean = []
        y_var = []
        y_sc_mean = []
        y_sc_var = []

        # Get the selected dataset
        for j in selected_index:
            # Do nothing for this point as it is not characterized, or it is a reference data point
            if (exp_data.iloc[j]["lsv_overpotential_-0.01"] == None) or ((repeat_exp[0:8] == exp_data.iloc[j][0:8]).all()):
                pass 
            else:
                # Append the x value first
                x_temp = [
                    (exp_data.iloc[j]["volume1"] - 0.1) / 8.5,
                    (exp_data.iloc[j]["volume2"] - 0.1) / 8.5,
                    (exp_data.iloc[j]["volume3"] - 0.1) / 8.5,
                    (exp_data.iloc[j]["volume4"] - 0.1) / 8.5,
                    (-(exp_data.iloc[j]["current"]) - 50) / 250,
                ]
                
                x.append(x_temp)
                
                # Deal with the current data
                if exp_data.iloc[j]["lsv_overpotential_-0.01"] != "unknown":
                    y_mean.append(1 * exp_data.iloc[j]["lsv_overpotential_-0.01"])
                else:
                    y_mean.append(-1.5)
                y_var.append((0.03**2))
                
                # Deal with the separation coefficient data
                y_sc_mean.append(1 * exp_data.iloc[j]["separation_coefficient"])
                y_sc_var.append((0.2**2))

        # Process the data input numpy array
        x = np.array(x).reshape(-1, 5)
        y_mean = np.array(y_mean).reshape(-1, 1)
        y_var = np.array(y_var).reshape(-1, 1)
        y_sc_mean = np.array(y_sc_mean).reshape(-1, 1)
        y_sc_var = np.array(y_sc_var).reshape(-1, 1)

        # If the data is not empty, perform machine learning
        if len(x) > 0:
            train_x = torch.tensor(x).reshape(-1, 5)
            train_y = torch.tensor(np.concatenate([y_mean + 1.5, y_sc_mean], axis=1))
            train_y_var = torch.tensor(np.concatenate([y_var, y_sc_var], axis=1))

            # Initialize the model
            mll, model = botorch_tool.tools.initialize_heter_noise_model(
                train_x=train_x, train_obj=train_y, train_obj_var=train_y_var
            )

            # Fit the model
            try:
                fit_gpytorch_mll(mll)
            except:
                print(f"Fitting noiseless model for combination {current_combo}")
                try:
                    mll, model = botorch_tool.tools.initialize_noiseless_model(
                        train_x=train_x, train_obj=train_y
                    )
                    fit_gpytorch_mll(mll)
                except:
                    raise ValueError(
                        f"Model fitting failed on combination {current_combo}!"
                    )

            # Get the bounds for the optimization
            bounds = torch.stack([torch.zeros(5), torch.ones(5)])

            # Sample the space according to the fitted model to perform BO
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=[0, 0],  # Use known reference point
                X_baseline=train_x,
                prune_baseline=True,
                sampler=sampler,
            )

            # Optimize the acquisition function
            candidate, expected_y = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=3,
                num_restarts=20,
                raw_samples=1024,
                inequality_constraints=[(torch.tensor([0, 1, 2, 3]), torch.tensor([1.0, 1.0, 1.0, 1.0]), 1.0)],
            )
            
            # Get three optimized samples
            print(f"{candidate} {expected_y}")

            # Convert to numpy format
            candidate_total = candidate.detach().cpu().numpy()
            candidate_rows_total = []

            for candidate in candidate_total:         
                # Get a new row for the candidate generation
                candidate_row = exp_data.iloc[exp_index].copy()
                
                volume1 = candidate[0] * 8.5 + 0.1
                volume2 = candidate[1] * 8.5 + 0.1
                volume3 = candidate[2] * 8.5 + 0.1
                volume4 = candidate[3] * 8.5 + 0.1
                volume5 = 9 - volume1 - volume2 - volume3 - volume4

                # Clip the values to ensure they stay within valid limits
                volume1 = np.clip(np.around(volume1, 1), 0.1, 8.6)
                volume2 = np.clip(np.around(volume2, 1), 0.1, 8.6)
                volume3 = np.clip(np.around(volume3, 1), 0.1, 8.6)
                volume4 = np.clip(np.around(volume4, 1), 0.1, 8.6)
                volume5 = np.clip(np.around(volume5, 1), 0.0, 8.6)  # Ensure non-negative volume for volume5

                candidate_row["volume1"] = volume1
                candidate_row["volume2"] = volume2
                candidate_row["volume3"] = volume3
                candidate_row["volume4"] = volume4
                candidate_row["volume5"] = volume5
                candidate_row["current"] = np.around(candidate[4] * 250 + 50, 0) * (-1)

                # Fill the rest of the fields
                candidate_row["syn_completed"] = "None"
                candidate_row["storage_location"] = "None"
                candidate_row["in_storage"] = "None"
                candidate_row["cat_completed"] = "None"
                candidate_row["lsv_overpotential_-0.01"] = "None"
                candidate_row["lsv_overpotential_-0.02"] = "None"
                candidate_row["lsv_overpotential_-0.03"] = "None"
                candidate_row["lsv_overpotential_-0.04"] = "None"
                candidate_row["lsv_overpotential_-0.05"] = "None"
                candidate_row["ir_value"] = "None"
                candidate_row["mass_time_start"] = "None"
                candidate_row["mass_time_end"] = "None"
                candidate_row["separation_coefficient"] = "None" 
                
                # Update the row 
                candidate_row = candidate_row.replace({np.nan: None, "None": None})
                
                # Append it to list 
                candidate_rows_total.append(candidate_row)
            
            return candidate_rows_total, repeat_exp

        else:
            return "not_ready", None
    else:
        return "not_ready", None
    

import numpy as np
import pandas as pd
import torch

def generate_new_exp_P_new(exp_data: pd.DataFrame, exp_index: int):
    """
    Generate a series of new experiments from the previous data and the exp_index for multi-objective optimization of metal phosphide catalysts.
    Args:
        exp_data: a pandas DataFrame that records the experiments.
        exp_index: the index of the experiments when we call generate_new_exp, which determines the categories of the reactants.
    Returns:
        The next sampling points or None.
    """
    # Set up the random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Get the current combination of the experiments
    current_combo = exp_data[["substance1", "substance2"]].iloc[exp_index]
    identical_index = (exp_data[["substance1", "substance2"]] == current_combo).all(axis=1)
    selected_index = np.where(identical_index == True)[0]
    
    # Set up the reference experiment  
    repeat_exp = pd.Series(
        {
            "substance1": current_combo["substance1"],
            "volume1": float(np.around(6 * 9 / 11, 1)),
            "substance2": current_combo["substance2"],
            "volume2": float(np.around(2 * 9 / 11, 1)),
            "substance3": "Na3Cit",
            "volume3": float(np.around(2 * 9 / 11, 1)),
            "substance4": "NaH2PO2",
            "volume4": float(np.around(1 * 9 / 11, 1)),
            "substance5": "water",
            "volume5": float(np.around(0.0, 1)),
            "current": int(-300),
            "syn_completed": "None",
            "storage_location": "None",
            "in_storage": "None",
            "cat_completed": "None",
            "lsv_overpotential_-0.01": "None",
            "lsv_overpotential_-0.02": "None",
            "lsv_overpotential_-0.03": "None",
            "lsv_overpotential_-0.04": "None",
            "lsv_overpotential_-0.05": "None",
            "ir_value": "None",
            "mass_time_start": "None",
            "mass_time_end": "None",
            "separation_coefficient": "None"
        }
    )
    repeat_exp = repeat_exp.replace({np.nan: None, "None": None})
    
    # Check if all the experiments are characterized for this specific combo
    if sum(exp_data.loc[identical_index]["lsv_overpotential_-0.01"].values != None) >= 8:
        # Read in the data
        x = []

        # Store the current data
        y_mean = []
        y_var = []
        y_sc_mean = []
        y_sc_var = []

        # Get the selected dataset
        for j in selected_index:
            # Do nothing for this point as it is not characterized, or it is a reference data point
            if (exp_data.iloc[j]["lsv_overpotential_-0.01"] == None) or ((repeat_exp[0:8] == exp_data.iloc[j][0:8]).all()):
                pass 
            else:
                # Append the x value first
                x_temp = [
                    (exp_data.iloc[j]["volume1"] - 0.1) / 8.5,
                    (exp_data.iloc[j]["volume2"] - 0.1) / 8.5,
                    (exp_data.iloc[j]["volume3"] - 0.1) / 8.5,
                    (exp_data.iloc[j]["volume4"] - 0.1) / 8.5,
                    (-(exp_data.iloc[j]["current"]) - 50) / 250,
                ]
                
                x.append(x_temp)
                
                # Deal with the current data
                if exp_data.iloc[j]["lsv_overpotential_-0.01"] != "unknown":
                    y_mean.append(1 * exp_data.iloc[j]["lsv_overpotential_-0.01"])
                else:
                    y_mean.append(-1.5)
                y_var.append((0.03**2))
                
                # Deal with the separation coefficient data
                y_sc_mean.append(1 * exp_data.iloc[j]["separation_coefficient"])
                y_sc_var.append((0.2**2))

        # Process the data input numpy array
        x = np.array(x).reshape(-1, 5)
        y_mean = np.array(y_mean).reshape(-1, 1)
        y_var = np.array(y_var).reshape(-1, 1)
        y_sc_mean = np.array(y_sc_mean).reshape(-1, 1)
        y_sc_var = np.array(y_sc_var).reshape(-1, 1)

        # If the data is not empty, perform machine learning
        if len(x) > 0:
            train_x = torch.tensor(x).reshape(-1, 5)
            train_y = torch.tensor(np.concatenate([y_mean + 1.5, y_sc_mean], axis=1))
            train_y_var = torch.tensor(np.concatenate([y_var, y_sc_var], axis=1))

            # Initialize the model
            mll, model = botorch_tool.tools.initialize_heter_noise_model(
                train_x=train_x, train_obj=train_y, train_obj_var=train_y_var
            )

            # Fit the model
            try:
                fit_gpytorch_mll(mll)
            except:
                print(f"Fitting noiseless model for combination {current_combo}")
                try:
                    mll, model = botorch_tool.tools.initialize_noiseless_model(
                        train_x=train_x, train_obj=train_y
                    )
                    fit_gpytorch_mll(mll)
                except:
                    raise ValueError(
                        f"Model fitting failed on combination {current_combo}!"
                    )

            # Get the bounds for the optimization
            bounds = torch.stack([torch.zeros(5), torch.ones(5)])

            # Sample the space according to the fitted model to perform BO
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=[0, 0],  # Use known reference point
                X_baseline=train_x,
                prune_baseline=True,
                sampler=sampler,
            )

            # Optimize the acquisition function
            candidate, expected_y = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=3,
                num_restarts=20,
                raw_samples=1024,
                inequality_constraints=[(torch.tensor([0, 1, 2, 3]), torch.tensor([1.0, 1.0, 1.0, 1.0]), 1.0)],
            )
            
            # Get three optimized samples
            print(f"{candidate} {expected_y}")

            # Convert to numpy format
            candidate_total = candidate.detach().cpu().numpy()
            candidate_rows_total = []

            for candidate in candidate_total:         
                # Get a new row for the candidate generation
                candidate_row = exp_data.iloc[exp_index].copy()
                
                # Calculate volumes and make sure total volume is <= 9 mL
                volume1 = candidate[0] * 8.5 + 0.1
                volume2 = candidate[1] * 8.5 + 0.1
                volume3 = candidate[2] * 8.5 + 0.1
                volume4 = candidate[3] * 8.5 + 0.1
                total_volume = volume1 + volume2 + volume3 + volume4

                # If total volume exceeds 9 mL, scale the volumes
                if total_volume > 9:
                    scale_factor = 9 / total_volume
                    volume1 *= scale_factor
                    volume2 *= scale_factor
                    volume3 *= scale_factor
                    volume4 *= scale_factor

                # Calculate remaining volume for volume5
                volume5 = 9 - volume1 - volume2 - volume3 - volume4
                volume5 = np.clip(np.around(volume5, 1), 0.0, 8.6)  # Ensure volume5 is non-negative and <= 8.6 mL

                # Clip and round volumes to valid ranges
                volume1 = np.clip(np.around(volume1, 1), 0.1, 8.6)
                volume2 = np.clip(np.around(volume2, 1), 0.1, 8.6)
                volume3 = np.clip(np.around(volume3, 1), 0.1, 8.6)
                volume4 = np.clip(np.around(volume4, 1), 0.1, 8.6)

                # Set the volumes in the candidate row
                candidate_row["volume1"] = volume1
                candidate_row["volume2"] = volume2
                candidate_row["volume3"] = volume3
                candidate_row["volume4"] = volume4
                candidate_row["volume5"] = volume5
                candidate_row["current"] = np.around(candidate[4] * 250 + 50, 0) * (-1)

                # Fill the rest of the fields
                candidate_row["syn_completed"] = "None"
                candidate_row["storage_location"] = "None"
                candidate_row["in_storage"] = "None"
                candidate_row["cat_completed"] = "None"
                candidate_row["lsv_overpotential_-0.01"] = "None"
                candidate_row["lsv_overpotential_-0.02"] = "None"
                candidate_row["lsv_overpotential_-0.03"] = "None"
                candidate_row["lsv_overpotential_-0.04"] = "None"
                candidate_row["lsv_overpotential_-0.05"] = "None"
                candidate_row["ir_value"] = "None"
                candidate_row["mass_time_start"] = "None"
                candidate_row["mass_time_end"] = "None"
                candidate_row["separation_coefficient"] = "None" 
                
                # Update the row 
                candidate_row = candidate_row.replace({np.nan: None, "None": None})
                
                # Append it to list 
                candidate_rows_total.append(candidate_row)
            
            return candidate_rows_total, repeat_exp

        else:
            return "not_ready", None
    else:
        return "not_ready", None
