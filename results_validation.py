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

DIR = "additional_datasets/"
POPSIZE = 40
NGEN = 40
regressor_SCS_path = "./regressors/regressor_SCS.joblib"
regressor_FTG_path = "./regressors/regressor_FTG.joblib"
df_path = "{}randomly_generated_configuration.csv".format(DIR)
df2_path = "{}configurations_improved_{}_{}.csv".format(DIR, POPSIZE, NGEN)
features = ["SCS", "FTG"]

def calculateRegression(df): 
    regressor_SCS = load(regressor_SCS_path)
    regressor_FTG = load(regressor_FTG_path)
    random_success_probability = regressor_SCS.predict(df) 
    random_muscle_fatigue      = regressor_FTG.predict(df)
    return (random_success_probability, random_muscle_fatigue)

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

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude


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
        regressors_input = pd.DataFrame([row])
        random_values    = calculateRegression(regressors_input)
        result_local     = pd.DataFrame(columns=result_df.columns)
        result_local[features] = [random_values]
        result_df              = pd.concat([result_df, result_local], ignore_index=True)

    df2 = pd.read_csv(df2_path)

    NSGAII_SCS_FTG_values = df2[features].to_numpy().flatten()
    random_SCS_FTG_values = result_df.to_numpy().flatten()

    #missing: effective size analysis over the two samples   
    res = wilcoxon(NSGAII_SCS_FTG_values, random_SCS_FTG_values)
    
    print(res.pvalue, res.statistic)
    print(VD_A(NSGAII_SCS_FTG_values, random_SCS_FTG_values))
