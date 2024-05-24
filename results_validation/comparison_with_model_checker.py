import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import precision_recall_fscore_support

new_columns = ["PRSCS_IMPROV", "SCS", "FTG_HUM_1_IMPROV", "FTG"]
columns = ["PRSCS", "SCS", "FTG_HUM_1", "FTG"]

df_generated = pd.read_csv("additional_datasets/improved_configurations/configurations_improved_20_20.csv")
df_model_checker_to_improve = pd.read_csv("additional_datasets/initial_configuration_to_improve_random.csv")
df_model_checker_improved = pd.read_csv("additional_datasets/DPa 3.csv")
ftg_hum_1_improv = df_model_checker_improved["FTG_HUM_1"].str.split(r'\+-', expand=True)[0].astype(float)

df_model_checker_improved = df_model_checker_improved.fillna({'FTG_HUM_1' : 0, "PRSCS_LOWER_BOUND" : 0.9, "PRSCS_UPPER_BOUND": 1})
#df_model_checker_improved = df_model_checker_improved[df_model_checker_improved['PRSCS_UPPER_BOUND'] != 0.0981446]

metrics_df                     = pd.DataFrame(columns=new_columns)
metrics_df["PRSCS_IMPROV"]     = df_model_checker_improved[["PRSCS_LOWER_BOUND", "PRSCS_UPPER_BOUND"]].mean(axis=1)
metrics_df["SCS"]              = df_generated["SCS"]
metrics_df["FTG_HUM_1_IMPROV"] = ftg_hum_1_improv
metrics_df["FTG"]              = df_generated["FTG"]

# Set your threshold value
threshold = 0.9

# Define a function to check if a value is above the threshold and return 1 or 0 accordingly
def check_threshold(value):
    if value > threshold:
        return 1
    else:
        return 0

# Apply the function to each value in 'col1' and 'col2' and create new columns with the results
metrics_df['MC_above_threshold'] = metrics_df['PRSCS_IMPROV'].apply(lambda x: check_threshold(x))
metrics_df['NSGAII_above_threshold'] = metrics_df['SCS'].apply(lambda x: check_threshold(x))

metrics_df.to_csv(f"results_validation/validation_table_{threshold}.csv", index=False)

y_true = np.array(metrics_df["MC_above_threshold"])
y_pred = np.array(metrics_df["NSGAII_above_threshold"])
print(precision_recall_fscore_support(y_true, y_pred, average='macro'))

exit()

tolerance = 0.2

# Calculate the range within the tolerance
lower_bound = metrics_df["PRSCS_IMPROV"] - metrics_df["PRSCS_IMPROV"] * tolerance
upper_bound = metrics_df["PRSCS_IMPROV"] + metrics_df["PRSCS_IMPROV"] * tolerance

# Count occurrences where column1 is within the tolerance range of column2
count_within_tolerance = ((metrics_df["SCS"] >= lower_bound) & (metrics_df["SCS"] <= upper_bound)).sum()


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
plt.savefig('results_validation/plots_and_tables/comparison_model_checker_improved_dropna.png')
plt.close()

import seaborn as sns

# Specify the columns you want to plot
columns_to_plot = ["PRSCS_IMPROV", "SCS", "FTG_HUM_1_IMPROV", "FTG"]

# Melt the DataFrame
melted_df = pd.melt(metrics_df[columns_to_plot], var_name='Objective Function', value_name='Values')

# Create a violin plot for all columns
plt.figure(figsize=(10, 6))  # Adjust size as needed
sns.violinplot(x='Objective Function', y='Values', data=melted_df)
plt.title('Comparison of Objective Functions Values')
plt.ylabel('Objective Functions Values')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.savefig('results_validation/plots_and_tables/comparison_model_checker_improved_violin_dropna.png')
plt.show()

