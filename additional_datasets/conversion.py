import pandas as pd

def reverseProcessDataframe(df):
    # Define reverse mappings for each categorical feature
    freewill_reverse_mapping = {0: "foc", 1: "distr", 2: "free"}
    age_reverse_mapping = {0: "y", 1: "e"}
    health_reverse_mapping = {0: "h", 1: "s", 2: "u"}

    # Define reverse transformations dictionary
    reverse_transformations = {
        0: {'HUM_1_FW': freewill_reverse_mapping,
            'HUM_2_FW': freewill_reverse_mapping,
            'HUM_1_AGE': age_reverse_mapping,
            'HUM_2_AGE': age_reverse_mapping,
            'HUM_1_STA': health_reverse_mapping,
            'HUM_2_STA': health_reverse_mapping}
    }

    def reverse_clean(dataset):
        X = dataset.copy()
        for column in dataset.columns:
            if column in reverse_transformations[0]:
                X[column] = X[column].replace(reverse_transformations[0][column])
        return X

    X_reversed = reverse_clean(df)
    return X_reversed

df = pd.read_csv("additional_datasets/initial_configuration_to_improve.csv")

column_mapping = {
    "PRGS" : "PROGRESS",
    "PRSCS_LB" : "PRSCS_LOWER_BOUND",
    "PRSCS_UB" : "PRSCS_UPPER_BOUND"
}

df.rename(columns=column_mapping, inplace=True)

df["PROGRESS"]=df["PROGRESS"].astype(int)
df["PSCS__TAU"]=df["PSCS__TAU"].astype(int)

# Function to join four columns into two
def join_columns(row):
    value1 = str(row["HUM_1_POS_X"])
    value2 = str(row["HUM_1_POS_Y"])
    value3 = str(row["HUM_2_POS_X"])
    value4 = str(row["HUM_2_POS_Y"])
    value5 = str(row["HUM_1_AGE"])
    value6 = str(row["HUM_1_STA"])
    value7 = str(row["HUM_2_AGE"])
    value8 = str(row["HUM_2_STA"])
    joined_column_1 = ', '.join([value1, value2])
    joined_column_2 = ', '.join([value3, value4])
    joined_column_3 = '/'.join([value5, value6])
    joined_column_4 = '/'.join([value7, value8])
    return joined_column_1, joined_column_2, joined_column_3, joined_column_4

df = reverseProcessDataframe(df)

# Apply the function to create new columns
df["HUM_1_POS"], df["HUM_2_POS"], df["HUM_1_FTG"], df["HUM_2_FTG"] = zip(*df.apply(join_columns, axis=1))

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
    "FTG_HUM_2"
    ]

# columns_to_add = ['PRSCS_LOWER_BOUND', 'PRSCS_UPPER_BOUND', 'FTG_HUM_1', 'FTG_HUM_2']
# for column in columns_to_add:
#     df[column] = '' 

df = df[new_columns_order]

df.to_csv("additional_datasets/df_model_checker_to_improve.csv", index=False)