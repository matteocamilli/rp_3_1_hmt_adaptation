import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from joblib import dump, load
from scipy.stats import wilcoxon
import itertools as it
from bisect import bisect_left
from typing import List
import numpy as np
import pandas as pd
import scipy.stats as ss
from pandas import Categorical
import lime
import lime.lime_tabular
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


DIR = "additional_datasets/"
POPSIZE = 20
NGEN = 20


regressor_SCS_path = "./regressors/regressor_SCS.joblib"
regressor_FTG_path = "./regressors/regressor_FTG.joblib"

combinations = ["20-20/random", "20-40/random", "40-20/random", "40-40/random",]

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

NSGA_datasets_paths = [
    "additional_datasets/improved_configurations/configurations_improved_20_20.csv",
    "additional_datasets/improved_configurations/configurations_improved_20_40.csv",
    "additional_datasets/improved_configurations/configurations_improved_40_20.csv",
    "additional_datasets/improved_configurations/configurations_improved_40_40.csv"
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

categorical_features = [
    "PRGS",
    "HUM_1_FW",
    "HUM_1_AGE",
    "HUM_1_STA",
    "HUM_2_FW",
    "HUM_2_AGE",
    "HUM_2_STA"
]

df_path = "{}configurations_to_improve/randomly_generated_configuration.csv".format(DIR)
df2_path = "{}improved_configurations/configurations_improved_{}_{}.csv".format(DIR, POPSIZE, NGEN)
features = ["SCS", "FTG"]
metrics_dataset_columns = ["Metric", "Configurations", "p-value", "Effect_size"]

def calculateRegression(df): 
    regressor_SCS = load(regressor_SCS_path)
    regressor_FTG = load(regressor_FTG_path)
    random_success_probability = regressor_SCS.predict(df)
    random_muscle_fatigue      = regressor_FTG.predict(df)
    return (random_success_probability.item(), random_muscle_fatigue.item())

def VD_A(treatment: List[float], control: List[float]):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000

    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/

    :param treatment: a numeric list
    :param control: another numeric list

    :returns the value estimate and the magnitude
    """
    m = len(treatment)
    n = len(control)

    if m != n:
        raise ValueError("Data d and f must have the same length")

    sample = np.concatenate([treatment, control])
    r = ss.rankdata(sample)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return magnitude

def VD_A_DF(data, val_col: str = None, group_col: str = None, sort=True):
    """

    :param data: pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.
    :param val_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains values.
    :param group_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains group names.
    :param sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.

    :return: stats : pandas DataFrame of effect sizes

    Stats summary ::
    'A' : Name of first measurement
    'B' : Name of second measurement
    'estimate' : effect sizes
    'magnitude' : magnitude

    """

    x = data.copy()
    if sort:
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)

    groups = x[group_col].unique()

    # Pairwise combinations
    g1, g2 = np.array(list(it.combinations(np.arange(groups.size), 2))).T

    # Compute effect size for each combination
    ef = np.array([VD_A(list(x[val_col][x[group_col] == groups[i]].values),
                        list(x[val_col][x[group_col] == groups[j]].values)) for i, j in zip(g1, g2)])

    return pd.DataFrame({
        'A': np.unique(data[group_col])[g1],
        'B': np.unique(data[group_col])[g2],
        'estimate': ef[:, 0],
        'magnitude': ef[:, 1]
    })

if __name__ == "__main__":
    df = pd.read_csv(df_path)
    result_df = pd.DataFrame(columns = features)
    
    for idx, row in df.iterrows(): 
        regressors_input       = pd.DataFrame([row])
        random_SCS, random_FTG = calculateRegression(regressors_input)
        result_local           = pd.DataFrame(columns=result_df.columns)
        result_local["SCS"]    = [random_SCS]
        result_local["FTG"]    = [random_FTG]
        result_df              = pd.concat([result_df, result_local], ignore_index=True)
    random_SCS_FTG_values = result_df
    
    NSGAII_values = []
    for file_name in NSGA_datasets_paths:
        df = pd.read_csv(file_name)
        NSGAII_values.append(df[features])

    final_df = pd.DataFrame(columns = metrics_dataset_columns)
    for j in features:
        for i in range(4):
            df_local = pd.DataFrame(columns=final_df.columns)
            res = wilcoxon(NSGAII_values[i][j], random_SCS_FTG_values[j])
            df_local.loc[0, "Metric"]  = j
            df_local["Configurations"] = combinations[i]
            df_local["p-value"]        = res.pvalue
            df_local["Effect_size"]    = VD_A(NSGAII_values[i][j], random_SCS_FTG_values[j])
            final_df                   = pd.concat([final_df, df_local], ignore_index=True)
    final_df.to_csv("results_validation/plots_and_tables_NSGAII/metrics_table.csv", index=False)    

    for idx, i in enumerate(NSGAII_values):
        plt.boxplot([i["SCS"]], positions=[idx+1], widths=0.6) 
    random_SCS_values = random_SCS_FTG_values["SCS"]
    plt.boxplot([random_SCS_values], positions=[5], widths=0.6) 
    plt.title('Distributions of Pscs')
    plt.ylabel('Pscs')
    plt.xlabel('Types of configuration generation')
    plt.xticks([1, 2, 3, 4, 5], ['20-20', '20-40', '40-20', '40-40', "random"])
    plt.savefig('results_validation/plots_and_tables_NSGAII/SCS_boxplot.png')
    plt.close()

    for idx, i in enumerate(NSGAII_values):
        plt.boxplot([i["FTG"]], positions=[idx+1], widths=0.6) 
    random_FTG_values = random_SCS_FTG_values["FTG"]
    plt.boxplot([random_FTG_values], positions=[5], widths=0.6) 
    plt.title('Distributions of FTG')
    plt.ylabel('FTG')
    plt.xlabel('Types of configuration generation')
    plt.xticks([1, 2, 3, 4, 5], ['20-20', '20-40', '40-20', '40-40', "random"])
    plt.savefig('results_validation/plots_and_tables_NSGAII/FTG_boxplot.png')
    plt.close()