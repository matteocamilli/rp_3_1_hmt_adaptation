import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from tqdm import tqdm

SUBSET = 1000
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
    "HUM_1_VEL",
    "HUM_2_VEL",
    "ROB_1_VEL"
]

constant_parameters = [
    "PRGS",
    "PSCS__TAU",
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


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["bestSCS"] = []
        self.data["bestFTG"] = []

    def notify(self, algorithm):
        self.data["bestSCS"].append((algorithm.pop.get("F")[:, [0]] * (-1)).mean())
        self.data["bestFTG"].append((algorithm.pop.get("F")[:, [1]]).mean())


class MOO(ElementwiseProblem):
    def __init__(self, row_unmodifiable, regressor_SCS_path, regressor_FTG_path, **kwargs):
        super().__init__(n_var=len(feature_names),
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([5.0, 2.0, 0.5, 0.1, 30.0, 30.0, 30.0]),  # Lower bounds for decision variables
                         xu=np.array([7.5, 4.5, 0.8, 0.4, 100.0, 100.0, 100.0]),
                         **kwargs)  # Upper bounds for decision variables

        self.row_unmodifiable = row_unmodifiable
        self.regressor_SCS = load(regressor_SCS_path)
        self.regressor_FTG = load(regressor_FTG_path)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.array(X).reshape(1, 7)
        model_input = self.reorganize_input_indices(X, self.row_unmodifiable)
        success_probability = self.regressor_SCS.predict(model_input)
        muscle_fatigue = self.regressor_FTG.predict(model_input)
        success_probability = -success_probability
        out["F"] = np.array([success_probability, muscle_fatigue])

    def reorganize_input_indices(self, row_modifiable, row_unmodifiable):
        new_df = pd.DataFrame(columns=all_features)
        new_df[feature_names] = row_modifiable
        new_df[constant_parameters] = row_unmodifiable
        return new_df


def processDataframe(df):
    freewill_mapping = {
        "foc": 0,
        "distr": 1,
        "free": 2
    }

    age_mapping = {
        "y": 0,
        "e": 1
    }

    health_mapping = {
        "h": 0,
        "s": 1,
        "u": 2
    }

    transformations = {
        'HUM_1_FW': freewill_mapping,
        'HUM_2_FW': freewill_mapping,
        'HUM_1_AGE': age_mapping,
        'HUM_2_AGE': age_mapping,
        'HUM_1_STA': health_mapping,
        'HUM_2_STA': health_mapping
    }

    def clean(dataset):
        X = dataset[all_features]
        for t in transformations:
            X = X.replace({t: transformations[t]})
        return X

    X = clean(df)
    return X


if __name__ == "__main__":
    df = pd.read_csv("additional_datasets/configurations_to_improve/initial_configurations_to_improve.csv")
    df = processDataframe(df)
    time_df = pd.DataFrame(columns=["Iteration_duration", "PSCS__TAU"])

    result_df = pd.DataFrame(columns=result_df_columns)

    val_SCS_averaged = []
    val_FTG_averaged = []

    unfeasible_configurations = 0

    for idx, (_, row) in tqdm(enumerate(df.iterrows()), total=df.shape[0]):
        iteration_start_time = time.time()

        problem = MOO(
            df[constant_parameters].to_numpy()[idx].reshape((1, len(constant_parameters))),
            "./regressors/regressor_SCS.joblib",
            "./regressors/regressor_FTG.joblib"
        )
        pop_size = int(sys.argv[1])
        algorithm = NSGA2(pop_size=pop_size)

        # Define the termination criteria
        termination = ("n_gen", int(sys.argv[2]))

        # Run the optimization
        res = minimize(problem,
                       algorithm,
                       termination=termination,
                       seed=1,
                       save_history=True,
                       callback=MyCallback(),
                       verbose=False)

        iteration_end_time = time.time()

        valSCS = res.algorithm.callback.data["bestSCS"]
        valFTG = res.algorithm.callback.data["bestFTG"]
        val_SCS_averaged.append(valSCS)
        val_FTG_averaged.append(valFTG)

        result_local = pd.DataFrame(columns=result_df.columns)
        result_local[feature_names] = res.X[-1].reshape((1, len(feature_names)))
        result_local[constant_parameters] = df[constant_parameters].to_numpy()[idx]
        result_local["SCS"] = -res.F[-1, 0]
        result_local["FTG"] = res.F[-1, 1]
        result_df = pd.concat([result_df, result_local], ignore_index=True)

        iteration_duration = iteration_end_time - iteration_start_time

        if (iteration_duration > df["PSCS__TAU"][idx]):
            unfeasible_configurations += 1

        time_df_local = pd.DataFrame(columns=time_df.columns)
        time_df_local.at[0, "Iteration_duration"] = iteration_duration
        time_df_local.at[0, "PSCS__TAU"] = df["PSCS__TAU"][idx]
        time_df = pd.concat([time_df, time_df_local], ignore_index=True)

    time_df.to_csv("additional_datasets/execution_time_log.csv", index=False)
    result_df.to_csv(
        f"additional_datasets/improved_configurations/configurations_improved_{termination[1]}_{pop_size}.csv",
        index=False)
    val_SCS_averaged = np.mean(val_SCS_averaged, axis=0)
    val_FTG_averaged = np.mean(val_FTG_averaged, axis=0)
    plt.plot(np.arange(len(val_SCS_averaged)), val_SCS_averaged, label="SCS")
    plt.plot(np.arange(len(val_FTG_averaged)), val_FTG_averaged, label="FTG")
    plt.xlabel('Number of generation')
    plt.ylabel('Objectives functions values')
    plt.title(f'Distributions of solution for n_gen={termination[1]}, pop_size={pop_size}')
    plt.legend()
    plt.savefig(f'results_validation/plots_and_tables_NSGAII/distributions_{termination[1]}_{pop_size}_averaged.png')
    plt.close()
