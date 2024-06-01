import pandas as pd

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

column_mapping = {
    "PROGRESS" : "PRGS",
    "PRSCS_LOWER_BOUND" : "PRSCS_LB",
    "PRSCS_UPPER_BOUND" : "PRSCS_UB"
}



df1 = pd.read_csv("additional_datasets/configurations_to_improve/initial_configurations_to_improve.csv")

df1.reset_index(drop=True, inplace=True)
df1.rename(columns=column_mapping, inplace=True)

df1 = df1[all_features]


df1.to_csv("additional_datasets/configurations_to_improve/initial_configurations_to_improve.csv")