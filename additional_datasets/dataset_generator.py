import pandas as pd
import random
import numpy as np

DIR = "data/ecsa2023ext/"
POINTS = 1000
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
]

df = pd.read_csv("{}dataset{}.csv".format(DIR, POINTS))
df = df.loc[(df['FTG_HUM_1'] >= 0.2) & (df['SCS'] < 1)]

variables_domain = [(5.0, 7.5), (2.0, 4.5), (0.5, 0.8), (0.1, 0.4), (250.0, 700.0), (30.0, 100.0), (30.0, 100.0), (30.0, 100.0)]
result_df = pd.DataFrame(columns=result_df_columns)

def processDataframe(df):
    freewill_mapping = {
    "foc" : 0,
    "distr" : 1,
    "free" : 2
    }

    age_mapping = {
        "y" : 0,
        "e" : 1
    }

    health_mapping = {
        "h" : 0,
        "s" : 1,
        "u" : 2
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
        X = dataset[result_df_columns]
        for t in transformations:
            X = X.replace({t: transformations[t]})
        return X

    X = clean(df)
    return X

for idx, (_, row) in enumerate(df.iterrows()):
    random_generated_variables = []
    for start, end in variables_domain:
        random_generated_variables.append(random.uniform(start, end))
    random_generated_variables = np.array(random_generated_variables).reshape((1, len(feature_names)))

    result_local = pd.DataFrame(columns=result_df.columns)
    result_local[feature_names] = random_generated_variables
    result_local[constant_parameters] = df[constant_parameters].to_numpy()[idx]
    result_df = pd.concat([result_df, result_local], ignore_index=True)

result_df = processDataframe(result_df)
result_df.to_csv("additional_datasets/randomly_generated_configuration.csv", index=False)
df.to_csv("additional_datasets/initial_configuration_to_improve.csv", index=False)


