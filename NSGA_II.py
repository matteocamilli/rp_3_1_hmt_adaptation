import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from joblib import dump, load
from multiprocessing.pool import Pool
from pymoo.core.problem import StarmapParallelization
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

DIR = "data/ecsa2023ext/"

SUBSET = 20
POINTS = 1000

all_features = [
    "PRGS",
    "ORCH_1_Dstop",
    "ORCH_1_Drestart",
    "ORCH_1_Fstop",
    "ORCH_1_Frestart",
    "PSCS__TAU",
    "HUM_1_VEL",
    "HUM_2_VEL",
    "HUM_1_FW",
    "HUM_1_AGE",
    "HUM_1_STA",
    "HUM_2_FW",
    "HUM_2_AGE",
    "HUM_2_STA",
    "HUM_1_POS_X",
    "HUM_1_POS_Y",
    "HUM_2_POS_X",
    "HUM_2_POS_Y",
    "ROB_1_VEL",
    "ROB_1_CHG"
]

feature_names = [ 
    "ORCH_1_Dstop",
    "ORCH_1_Drestart",
    "ORCH_1_Fstop",
    "ORCH_1_Frestart",
    "PSCS__TAU",
    "HUM_1_VEL",
    "HUM_2_VEL",
    "ROB_1_VEL"
]

constant_parameters = [
    "PRGS",
    "HUM_1_FW",
    "HUM_1_AGE",
    "HUM_1_STA",
    "HUM_2_FW",
    "HUM_2_AGE",
    "HUM_2_STA",
    "HUM_1_POS_X",
    "HUM_1_POS_Y",
    "HUM_2_POS_X",
    "HUM_2_POS_Y",
    "ROB_1_CHG",
]

result_df_columns = [
    "PRGS",
    "ORCH_1_Dstop",
    "ORCH_1_Drestart",
    "ORCH_1_Fstop",
    "ORCH_1_Frestart",
    "PSCS__TAU",
    "HUM_1_VEL",
    "HUM_2_VEL",
    "HUM_1_FW",
    "HUM_1_AGE",
    "HUM_1_STA",
    "HUM_2_FW",
    "HUM_2_AGE",
    "HUM_2_STA",
    "HUM_1_POS_X",
    "HUM_1_POS_Y",
    "HUM_2_POS_X",
    "HUM_2_POS_Y",
    "ROB_1_VEL",
    "ROB_1_CHG",
    "SCS",
    "FTG"
]

class MOO(ElementwiseProblem):
    def __init__(self, row_unmodifiable, regressor_SCS_path, regressor_FTG_path, **kwargs):
        super().__init__(n_var=len(feature_names),
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([5.0, 2.0, 0.5, 0.1, 250.0, 30.0, 30.0, 30.0]),  # Lower bounds for decision variables
                         xu=np.array([7.5, 4.5, 0.8, 0.4, 700.0, 100.0, 100.0, 100.0 ]), **kwargs)  # Upper bounds for decision variables
        
        self.row_unmodifiable         = row_unmodifiable
        self.regressor_SCS            = load(regressor_SCS_path)
        self.regressor_FTG            = load(regressor_FTG_path)
    
    def _evaluate(self, X, out, *args, **kwargs):
        X = np.array(X).reshape(1 , 8)
        model_input         = self.reorganize_input_indices(X, self.row_unmodifiable)
        success_probability = self.regressor_SCS.predict(model_input) 
        muscle_fatigue      = self.regressor_FTG.predict(model_input)
        success_probability = -success_probability
        out["F"] = np.array([success_probability, muscle_fatigue])

    def reorganize_input_indices(self, row_modifiable, row_unmodifiable): 
        new_df = pd.DataFrame(columns=all_features)
        new_df[feature_names] = row_modifiable
        new_df[constant_parameters] = row_unmodifiable
        return new_df
    
if __name__ == "__main__":
        
    ## Load initial dataset
    df = pd.read_csv("{}dataset{}.csv".format(DIR, POINTS)).head(SUBSET)

    result_df = pd.DataFrame(columns=result_df_columns)
    n_proccess = 8

    class MyCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.data["best"] = []

        def notify(self, algorithm):
            self.data["best"].append(algorithm.pop.get("F").mean())

    for idx, (_, row) in tqdm(enumerate(df.iterrows()), total=df.shape[0]): 
        # pool = Pool(n_proccess)
        # runner = StarmapParallelization(pool.starmap)
        
        problem = MOO( 
            df[constant_parameters].to_numpy()[idx].reshape((1, len(constant_parameters))),
            "./regressors/regressor_SCS.joblib",
            "./regressors/regressor_FTG.joblib", 
            #elementwise_runner=runner,
        )
        pop_size = 100
        algorithm = NSGA2(pop_size=pop_size)

        # Define the termination criteria
        termination = ("n_gen", 20)

        # Run the optimization
        res = minimize(problem,
                    algorithm,
                    termination=termination,
                    seed=1,
                    save_history=True,
                    callback=MyCallback(),
                    verbose=False)

        #Print the results
        # print("Best solution found:")
        # print("Success Probability:", -res.F[-1, 0])
        # print("Muscle Fatigue:", res.F[-1, 1]) 
        # print("Old solution: ", df[feature_names].to_numpy()[idx])
        # print("New solution: ", res.X[-1])

        #Plotting some stuffs 
        val = res.algorithm.callback.data["best"]
        plt.plot(np.arange(len(val)), val)
        plt.xlabel('Number of generation')
        plt.ylabel('Objectives values')
        plt.title(f'Initial configuration number {idx}, population size: {pop_size}') 
        folder_path = 'plots_folder'
        plt.savefig(os.path.join(folder_path, f'plot_{idx}.png'))
        plt.close()

        #pool.close()  

        result_local = pd.DataFrame(columns=result_df.columns)
        result_local[feature_names] = res.X[-1].reshape((1, len(feature_names)))
        result_local[constant_parameters] = df[constant_parameters].to_numpy()[idx]
        result_local["SCS"] = -res.F[-1, 0]
        result_local["FTG"] = res.F[-1, 1]
        result_df = pd.concat([result_df, result_local], ignore_index=True)

        
    result_df.to_csv("dataset1000_improved", index=False)
        