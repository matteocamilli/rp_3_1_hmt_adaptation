import pandas as pd

df = pd.read_csv("additional_datasets/configurations_improved_20_20.csv")

column_mapping = {
    "PRGS" : "PROGRESS",
}

df.rename(columns=column_mapping, inplace=True)

# Function to join four columns into two
def join_columns(row):
    value1 = str(row["HUM_1_POS_X"])
    value2 = str(row["HUM_1_POS_Y"])
    value3 = str(row["HUM_2_POS_X"])
    value4 = str(row["HUM_2_POS_Y"])
    joined_column_1 = ', '.join([value1, value2])
    joined_column_2 = ', '.join([value2, value4])
    return joined_column_1, joined_column_2

# Apply the function to create new columns
df["HUM_1_POS"], df["HUM_2_POS"] = zip(*df.apply(join_columns, axis=1))

new_columns_order = [
    "PROGRESS", 
    "ORCH_1_Dstop", 
    "ORCH_1_Drestart", 
    "ORCH_1_Fstop", 
    "ORCH_1_Frestart", 
    "PSCS__TAU", 
    "HUM_1_VEL", 
    "HUM_2_VEL", 
    "HUM_1_FW", 
    "HUM_1_FTG", 
    "HUM_2_FW", 
    "HUM_2_FTG", 
    "HUM_1_POS", 
    "HUM_2_POS", 
    "ROB_1_VEL", 
    "ROB_1_CHG", 
    "PRSCS_LOWER_BOUND", 
    "PRSCS_UPPER_BOUND", 
    "FTG_HUM_1", 
    "FTG_HUM_2", 
    "SCS",
    "FTG"
    ]

columns_to_add = ['HUM_1_FTG', 'HUM_2_FTG', 'PRSCS_LOWER_BOUND', 'PRSCS_UPPER_BOUND', 'FTG_HUM_1', 'FTG_HUM_2']
for column in columns_to_add:
    df[column] = '' 

df = df[new_columns_order]

df.to_csv("additional_datasets/df_model_checker.csv", index=False)