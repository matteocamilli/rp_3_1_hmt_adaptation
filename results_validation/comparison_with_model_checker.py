import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

new_columns = ["PRSCS_INIT", "PRSCS_IMPROV", "SCS", "FTG_HUM_1_INIT", "FTG_HUM_1_IMPROV", "FTG"]
columns = ["PRSCS", "SCS", "FTG_HUM_1", "FTG"]

df_generated = pd.read_csv("additional_datasets/improved_configurations/configurations_improved_20_20.csv")
df_model_checker_to_improve = pd.read_csv("additional_datasets/initial_configuration_to_improve_random.csv")
df_model_checker_improved = pd.read_csv("additional_datasets/DPa 3.csv")
ftg_hum_1_improv = df_model_checker_improved["FTG_HUM_1"].str.split(r'\+-', expand=True)[0].astype(float)

df_model_checker_improved = df_model_checker_improved.fillna({'FTG_HUM_1' : 0, "PRSCS_LOWER_BOUND" : 0.9, "PRSCS_UPPER_BOUND": 1})


metrics_df = pd.DataFrame(columns=new_columns)
metrics_df["PRSCS_INIT"]       = df_model_checker_to_improve[["PRSCS_LOWER_BOUND", "PRSCS_UPPER_BOUND"]].mean(axis=1)
metrics_df["PRSCS_IMPROV"]     = df_model_checker_improved[["PRSCS_LOWER_BOUND", "PRSCS_UPPER_BOUND"]].mean(axis=1)
metrics_df["SCS"]              = df_generated["SCS"]
metrics_df["FTG_HUM_1_INIT"]   = df_model_checker_to_improve["FTG_HUM_1"]
metrics_df["FTG_HUM_1_IMPROV"] = ftg_hum_1_improv
metrics_df["FTG"]              = df_generated["FTG"]

# df_migliorate = pd.DataFrame(columns=columns)
# df_peggiorate = pd.DataFrame(columns=columns)

# for idx, row in metrics_df.iterrows():
#     local_df = pd.DataFrame(columns=columns)
#     local_df.at[0, "SCS"] = row["SCS"]
#     local_df.at[0, "FTG"] = row["FTG"]
#     local_df.at[0, "PRSCS"] = row["PRSCS_IMPROV"]
#     local_df.at[0, "FTG_HUM_1"] = row["FTG_HUM_1_IMPROV"]
    
#     if row["SCS"] > row["PRSCS_INIT"]:
#         df_migliorate = pd.concat([df_migliorate, local_df], ignore_index=True)
#     else:
#         df_peggiorate = pd.concat([df_peggiorate, local_df], ignore_index=True)

metrics_df = metrics_df.fillna({'FTG_HUM_1_IMPROV' : 0, "PRSCS_IMPROV" : 0.9})

plt.boxplot([metrics_df["PRSCS_IMPROV"]], positions=[1], widths=0.6) 
plt.boxplot([metrics_df["SCS"]], positions=[2], widths=0.6) 
plt.boxplot([metrics_df["FTG_HUM_1_IMPROV"]], positions=[3], widths=0.6)
plt.boxplot([metrics_df["FTG"]], positions=[4], widths=0.6) 
plt.title('Comparison of Objective Functions Values')
plt.ylabel('Objective Functions Values')
plt.xlabel('Method of prediction of a specific Objective Function')
plt.xticks([1, 2, 3, 4], ['SCS - MC', 'SCS - NSGAII', 'FTG - MC', 'FTG - NSGAII'])
plt.savefig('results_validation/plots_and_tables/comparison_model_checker_improved.png')
plt.close()
