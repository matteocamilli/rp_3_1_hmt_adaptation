import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

scs_columns = ["SCS_INITIAL", "SCS_NSGA_II_20_20", "SCS_NSGA_II_20_40", "SCS_NSGA_II_40_20", "SCS_NSGA_II_40_40",
               "SCS_RND"]
ftg_columns = ["FTG_INITIAL", "FTG_NSGA_II_20_20", "FTG_NSGA_II_20_40", "FTG_NSGA_II_40_20", "FTG_NSGA_II_40_40",
               "FTG_RND"]

df_to_improve = pd.read_csv("additional_datasets/configurations_to_improve/"
                            "initial_with_regressor_estimations.csv")
df_improved_20_20 = pd.read_csv("additional_datasets/improved_configurations/configurations_improved_20_20.csv")
df_improved_20_40 = pd.read_csv("additional_datasets/improved_configurations/configurations_improved_20_40.csv")
df_improved_40_20 = pd.read_csv("additional_datasets/improved_configurations/configurations_improved_40_20.csv")
df_improved_40_40 = pd.read_csv("additional_datasets/improved_configurations/configurations_improved_40_40.csv")
df_random = pd.read_csv("additional_datasets/configurations_to_improve/"
                        "randomly_generated_configurations.csv")

scs_metrics_df = pd.DataFrame(columns=scs_columns)
# Uncomment to plot SMC estimations on initial configurations
# scs_metrics_df["SCS_INITIAL"] = df_to_improve[["PRSCS_LB", "PRSCS_UB"]].mean(axis=1)
# Uncomment to plot regressor estimations on initial configurations
scs_metrics_df["SCS_INITIAL"] = df_to_improve["SCS"].values
scs_metrics_df["SCS_NSGA_II_20_20"] = df_improved_20_20["SCS"].values
scs_metrics_df["SCS_NSGA_II_20_40"] = df_improved_20_40["SCS"].values
scs_metrics_df["SCS_NSGA_II_40_20"] = df_improved_40_20["SCS"].values
scs_metrics_df["SCS_NSGA_II_40_40"] = df_improved_40_40["SCS"].values
scs_metrics_df["SCS_RND"] = df_random["SCS"].values

ftg_metrics_df = pd.DataFrame(columns=ftg_columns)
ftg_metrics_df["FTG_INITIAL"] = df_to_improve["FTG"]
ftg_metrics_df["FTG_NSGA_II_20_20"] = df_improved_20_20["FTG"]
ftg_metrics_df["FTG_NSGA_II_20_40"] = df_improved_20_40["FTG"]
ftg_metrics_df["FTG_NSGA_II_40_20"] = df_improved_40_20["FTG"]
ftg_metrics_df["FTG_NSGA_II_40_40"] = df_improved_40_40["FTG"]
ftg_metrics_df["FTG_RND"] = df_random["FTG"]

# Set your threshold value
threshold = 0.9


# Define a function to check if a value is above the threshold and return 1 or 0 accordingly
def check_threshold(value):
    if value > threshold:
        return 1
    else:
        return 0


# Apply the function to each value in 'col1' and 'col2' and create new columns with the results
scs_metrics_df['MC_above_threshold'] = scs_metrics_df['SCS_INITIAL'].apply(lambda x: check_threshold(x))
scs_metrics_df['NSGAII_above_threshold'] = scs_metrics_df['SCS_NSGA_II_20_20'].apply(lambda x: check_threshold(x))

scs_metrics_df.to_csv(f"results_validation/comparison_with_MC/validation_table_{threshold}.csv", index=False)

y_true = np.array(scs_metrics_df["MC_above_threshold"])
y_pred = np.array(scs_metrics_df["NSGAII_above_threshold"])
print(precision_recall_fscore_support(y_true, y_pred, average='macro'))

tolerance = 0.2

# Calculate the range within the tolerance
lower_bound = scs_metrics_df["SCS_INITIAL"] - scs_metrics_df["SCS_INITIAL"] * tolerance
upper_bound = scs_metrics_df["SCS_INITIAL"] + scs_metrics_df["SCS_INITIAL"] * tolerance

count_within_tolerance = ((scs_metrics_df["SCS_NSGA_II_20_20"] >= lower_bound) & (
        scs_metrics_df["SCS_NSGA_II_20_20"] <= upper_bound)).sum()

scs_metrics_df = scs_metrics_df.fillna({"SCS_INITIAL": 0.9})
ftg_metrics_df = ftg_metrics_df.fillna({'FTG_INITIAL': 0})

fig, ax = plt.subplots(ncols=2, figsize=(20, 6))

ax[0].boxplot([scs_metrics_df["SCS_INITIAL"]], positions=[1], widths=0.6)
ax[0].boxplot([scs_metrics_df["SCS_RND"]], positions=[2], widths=0.6)
ax[0].boxplot([scs_metrics_df["SCS_NSGA_II_20_20"]], positions=[3], widths=0.6)
ax[0].boxplot([scs_metrics_df["SCS_NSGA_II_20_40"]], positions=[4], widths=0.6)
ax[0].boxplot([scs_metrics_df["SCS_NSGA_II_40_20"]], positions=[5], widths=0.6)
ax[0].boxplot([scs_metrics_df["SCS_NSGA_II_40_40"]], positions=[6], widths=0.6)

ax[1].boxplot([ftg_metrics_df["FTG_INITIAL"]], positions=[7], widths=0.6)
ax[1].boxplot([ftg_metrics_df["FTG_RND"]], positions=[8], widths=0.6)
ax[1].boxplot([ftg_metrics_df["FTG_NSGA_II_20_20"]], positions=[9], widths=0.6)
ax[1].boxplot([ftg_metrics_df["FTG_NSGA_II_20_40"]], positions=[10], widths=0.6)
ax[1].boxplot([ftg_metrics_df["FTG_NSGA_II_40_20"]], positions=[11], widths=0.6)
ax[1].boxplot([ftg_metrics_df["FTG_NSGA_II_40_40"]], positions=[12], widths=0.6)

ax[0].set_title('Success Probability')
ax[0].set_xticks([1, 2, 3, 4, 5, 6], ['Initial', 'RAND', 'NSGA-II (20, 20)', 'NSGA-II (20, 40)',
                                      'NSGA-II (40, 20)', 'NSGA-II (40, 40)'])
ax[1].set_title('Fatigue Level')
ax[1].set_xticks([7, 8, 9, 10, 11, 12], ['Initial', 'RAND', 'NSGA-II (20, 20)', 'NSGA-II (20, 40)',
                                         'NSGA-II (40, 20)', 'NSGA-II (40, 40)'])

plt.savefig('results_validation/comparison_with_MC/comparison_model_checker_improved_dropna.png', bbox_inches='tight')
plt.show()
plt.close()

# Specify the columns you want to plot
# columns_to_plot = ["SCS_INITIAL", "SCS_NSGA_II", "SCS_RND", "FTG_INITIAL", "FTG_NSGA_II", "FTG_RND"]

# Melt the DataFrame
# melted_df = pd.melt(metrics_df[columns_to_plot], var_name='Objective Function', value_name='Values')

# Create a violin plot for all columns
# plt.figure(figsize=(10, 6))  # Adjust size as needed
# sns.violinplot(x='Objective Function', y='Values', data=melted_df)
# plt.title('Comparison of Objective Functions Values')
# plt.ylabel('Objective Functions Values')
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
# plt.tight_layout()
# plt.savefig('results_validation/comparison_with_MC/comparison_model_checker_improved_violin_dropna.png')
