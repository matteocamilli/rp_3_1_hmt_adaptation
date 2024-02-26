# Classifiers
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # Neural Network
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
# AUC calculation
from sklearn.metrics import roc_auc_score
# Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, chi2

from joblib import dump, load

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

DIR = "data/ecsa2023ext/"

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

class MOO(Problem):
    def __init__(self, row_modifiable, row_unmodifiable, row_modifiable_idx_map, row_unmodifiable_idx_map, classifier_path, regressor_path,population_size,  max_variation = 0.1):
        super().__init__(n_var=len(feature_names),
                         n_obj=2,
                         n_constr=0,  #should i add constraints for every single decision variables and also for the prediction?
                         xl=np.array([(1 - max_variation) * val for val in row_modifiable]),  # Lower bounds for decision variables
                         xu=np.array([(1 + max_variation) * val for val in row_modifiable]))  # Upper bounds for decision variables
        
        self.row_unmodifiable         = np.repeat(row_unmodifiable, population_size, axis=0)
        self.row_modifiable_idx_map   = row_modifiable_idx_map
        self.row_unmodifiable_idx_map = row_unmodifiable_idx_map
        self.my_classifier            = load(classifier_path)
        self.my_regressor             = load(regressor_path)
    
    def _evaluate(self, X, out, *args, **kwargs):
        model_input         = self.reorganize_input_indices(X, self.row_unmodifiable)
        success_probability = self.my_classifier.predict(model_input) 
        muscle_fatigue      = self.my_regressor.predict(model_input)
        success_probability = -success_probability
        
        # Storing of the results
        out["F"] = np.array([success_probability, muscle_fatigue])

    def reorganize_input_indices(self, row_modifiable, row_unmodifiable): 
        new_df = pd.DataFrame(columns=all_features)

        for (idx, fn_name) in self.row_modifiable_idx_map.items(): 
            new_df[fn_name] = row_modifiable[:, idx]

        for (idx, cp_name) in self.row_unmodifiable_idx_map.items(): 
            new_df[cp_name] = row_unmodifiable[:, idx]

        #print(new_df.iloc[0])
        return new_df
    
if __name__ == "__main__":
        
    ## Load initial dataset
    df = pd.read_csv("{}dataset{}.csv".format(DIR, POINTS)).head(SUBSET)

    ## Filter out all the variables that we should use as decision variables
    df_decision_vars = df[feature_names]

    row_modifiable_idx_map = {}
    row_unmodifiable_idx_map = {}

    for i, fn in enumerate(feature_names): 
        row_modifiable_idx_map[i] = fn

    for i, cp in enumerate(constant_parameters): 
        row_unmodifiable_idx_map[i] = cp

    result_df = pd.DataFrame(columns=feature_names)
    population_size = 100

    for idx, (_, row) in tqdm(enumerate(df.iterrows()), total=df.shape[0]): 
        problem = MOO(
            df[feature_names].to_numpy()[idx], 
            df[constant_parameters].to_numpy()[idx].reshape((1, len(constant_parameters))), 
            row_modifiable_idx_map,
            row_unmodifiable_idx_map,
            "./classifier.joblib",
            "./regressor.joblib", 
            population_size
        )

        algorithm = NSGA2(pop_size=population_size)

        # Define the termination criteria
        termination = ("n_gen", 100)

        # Run the optimization
        res = minimize(problem,
                    algorithm,
                    termination=termination,
                    seed=1,
                    save_history=True,
                    verbose=False)

        # Print the results
        # print("Best solution found:")
        # print("Success Probability:", res.F[-1, 0])
        # print("Muscle Fatigue:", res.F[-1, 1]) 
        # print("Old solution: ", df[feature_names].to_numpy()[idx])
        # print("New solution: ", res.X[-1])
        result_local = pd.DataFrame(res.X[-1].reshape((1, len(result_df.columns))), columns=result_df.columns)
        result_df = pd.concat([result_df, result_local], ignore_index=True)

    result_df.to_csv("dataset1000_improved.csv", index=False)
        