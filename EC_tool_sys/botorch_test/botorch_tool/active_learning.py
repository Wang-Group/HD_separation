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


    
