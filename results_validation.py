import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from joblib import dump, load

regressor_SCS_path = "./regressors/regressor_SCS.joblib"
regressor_FTG_path = "./regressors/regressor_FTG.joblib"
df_path = "additional_datasets/randomly_generated_configuration.csv"
df2_path = "initial_configurations_improved.csv"
features = ["SCS", "FTG"]

def calculateRegression(df): 
    regressor_SCS = load(regressor_SCS_path)
    regressor_FTG = load(regressor_FTG_path)
    random_success_probability = regressor_SCS.predict(df) 
    random_muscle_fatigue      = regressor_FTG.predict(df)
    return (random_success_probability, random_muscle_fatigue)

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
    
    NSGAII_SCS_FTG_values = df2[features]
    random_SCS_FTG_values = result_df
