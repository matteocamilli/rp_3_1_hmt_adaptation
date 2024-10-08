import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

new_columns = ["SCS_INITIAL", "SCS_NSGA_II", "SCS_RND", "FTG_INITIAL", "FTG_NSGA_II", "FTG_RND"]

df_to_improve = pd.read_csv("additional_datasets/configurations_to_improve/"
                            "initial_configurations_to_improve.csv")
df_improved = pd.read_csv("additional_datasets/improved_configurations/configurations_improved_20_20.csv")
df_random = pd.read_csv("additional_datasets/configurations_to_improve/"
                        "randomly_generated_configurations.csv")

metrics_df = pd.DataFrame(columns=new_columns)
metrics_df["SCS_INITIAL"] = df_to_improve[["PRSCS_LB", "PRSCS_UB"]].mean(axis=1)
metrics_df["SCS_NSGA_II"] = df_improved["SCS"].values
metrics_df["SCS_RND"] = df_random["SCS"].values
metrics_df["FTG_INITIAL"] = df_to_improve["FTG_HUM_1"]
metrics_df["FTG_NSGA_II"] = df_improved["FTG"]
metrics_df["FTG_RND"] = df_random["FTG"]

# Set your threshold value
threshold = 0.9


# Define a function to check if a value is above the threshold and return 1 or 0 accordingly
def check_threshold(value):
    if value > threshold:
        return 1
    else:
        return 0


# Apply the function to each value in 'col1' and 'col2' and create new columns with the results
metrics_df['MC_above_threshold'] = metrics_df['SCS_INITIAL'].apply(lambda x: check_threshold(x))
metrics_df['NSGAII_above_threshold'] = metrics_df['SCS_NSGA_II'].apply(lambda x: check_threshold(x))

metrics_df.to_csv(f"results_validation/comparison_with_MC/validation_table_{threshold}.csv", index=False)

y_true = np.array(metrics_df["MC_above_threshold"])
y_pred = np.array(metrics_df["NSGAII_above_threshold"])
print(precision_recall_fscore_support(y_true, y_pred, average='macro'))

tolerance = 0.2

# Calculate the range within the tolerance
lower_bound = metrics_df["SCS_INITIAL"] - metrics_df["SCS_INITIAL"] * tolerance
upper_bound = metrics_df["SCS_INITIAL"] + metrics_df["SCS_INITIAL"] * tolerance

count_within_tolerance = ((metrics_df["SCS_NSGA_II"] >= lower_bound) & (metrics_df["SCS_NSGA_II"] <= upper_bound)).sum()

metrics_df = metrics_df.fillna({'FTG_INITIAL': 0, "SCS_INITIAL": 0.9})

fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

ax[0].boxplot([metrics_df["SCS_INITIAL"]], positions=[1], widths=0.6)
ax[0].boxplot([metrics_df["SCS_NSGA_II"]], positions=[2], widths=0.6)
ax[0].boxplot([metrics_df["SCS_RND"]], positions=[3], widths=0.6)
ax[1].boxplot([metrics_df["FTG_INITIAL"]], positions=[4], widths=0.6)
ax[1].boxplot([metrics_df["FTG_NSGA_II"]], positions=[5], widths=0.6)
ax[1].boxplot([metrics_df["FTG_RND"]], positions=[6], widths=0.6)

ax[0].set_title('Success Probability')
ax[0].set_xticks([1, 2, 3], ['Initial', 'NSGA-II', 'RAND'])
ax[1].set_title('HUM1 Fatigue Level')
ax[1].set_xticks([4, 5, 6], ['Initial', 'NSGA-II', 'RAND'])

plt.savefig('results_validation/comparison_with_MC/comparison_model_checker_improved_dropna.png', bbox_inches='tight')
plt.show()
plt.close()

# Specify the columns you want to plot
columns_to_plot = ["SCS_INITIAL", "SCS_NSGA_II", "SCS_RND", "FTG_INITIAL", "FTG_NSGA_II", "FTG_RND"]

# Melt the DataFrame
melted_df = pd.melt(metrics_df[columns_to_plot], var_name='Objective Function', value_name='Values')

# Create a violin plot for all columns
plt.figure(figsize=(10, 6))  # Adjust size as needed
sns.violinplot(x='Objective Function', y='Values', data=melted_df)
plt.title('Comparison of Objective Functions Values')
plt.ylabel('Objective Functions Values')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.savefig('results_validation/comparison_with_MC/comparison_model_checker_improved_violin_dropna.png')
