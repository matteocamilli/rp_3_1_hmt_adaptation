import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import sys

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

regressor_SCS_path = "./regressors/regressor_SCS_LIME.joblib"
regressor_FTG_path = "./regressors/regressor_FTG_LIME.joblib"

categorical_features = [
    "PRGS",
    "HUM_1_FW",
    "HUM_1_AGE",
    "HUM_1_STA",
    "HUM_2_FW",
    "HUM_2_AGE",
    "HUM_2_STA"
]

feature_names = [
    "ORCH_1_Dstop",
    "ORCH_1_Drestart",
    "ORCH_1_Fstop",
    "ORCH_1_Frestart",
    "HUM_1_VEL",
    "HUM_2_VEL",
    "ROB_1_VEL"
]

if __name__ == "__main__":
    #read the data
    df = pd.read_csv("additional_datasets/improved_configurations/configurations_improved_20_20.csv")
    X_train_SCS = pd.read_csv("regressors/X_train_SCS")
    X_train_FTG = pd.read_csv("regressors/X_train_FTG")

    # Initialize LimeTabularExplainer
    explainer_SCS = lime.lime_tabular.LimeTabularExplainer(X_train_SCS.values, 
                                                    feature_names=X_train_SCS.columns.values.tolist(), 
                                                    class_names=["SCS"], 
                                                    verbose=True, 
                                                    mode='regression')
    
    explainer_FTG = lime.lime_tabular.LimeTabularExplainer(X_train_FTG.values, 
                                                    feature_names=X_train_FTG.columns.values.tolist(), 
                                                    class_names=["FTG"], 
                                                    verbose=True, 
                                                    mode='regression')
    

    # Choose a specific data point to analyze
    i = 10
    df=df[all_features]
    data = df.iloc[[i]]
    data.reset_index(drop=True, inplace=True)
    
    #load the regressor
    regressor_SCS = load(regressor_SCS_path)
    regressor_FTG = load(regressor_FTG_path)

    # Explain the instance
    exp_SCS = explainer_SCS.explain_instance(df.values[i], regressor_SCS.predict, num_features=len(feature_names))
    exp_FTG = explainer_FTG.explain_instance(df.values[i], regressor_FTG.predict, num_features=len(feature_names))

    fig_SCS = exp_SCS.as_pyplot_figure()
    fig_SCS.set_size_inches(21.5, 15.5)

    fig_FTG = exp_FTG.as_pyplot_figure()
    fig_FTG.set_size_inches(21.5, 15.5)

    # Save the figures
    fig_SCS.savefig('results_validation/plots_and_tables/lime_explanation_SCS.png')
    fig_FTG.savefig('results_validation/plots_and_tables/lime_explanation_FTG.png')

    # Close the plots
    plt.close(fig_SCS)
    plt.close(fig_FTG)

