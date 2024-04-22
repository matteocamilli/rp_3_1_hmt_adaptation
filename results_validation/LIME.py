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

regressor_SCS_path = "./regressors/regressor_SCS.joblib"

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

if __name__ == "__main__":
    #read the data
    df = pd.read_csv("additional_datasets/configurations_improved_20_20.csv")
    X_train = pd.read_csv("regressors/X_train.csv")

    # Initialize LimeTabularExplainer
    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train), 
                                                    feature_names=X_train.columns.values.tolist(), 
                                                    #categorical_features=categorical_features, 
                                                    verbose=True, 
                                                    mode='regression')
    
    # Choose a specific data point to analyze
    i = 10
    
    #load the regressor
    regressor = load(regressor_SCS_path)
    
    # Explain the instance
    exp = explainer.explain_instance(df[all_features].iloc[[i]], regressor.predict, num_features=len(feature_names))
    fig = exp.as_pyplot_figure()
    fig.savefig('lime_explanation.png')
    plt.close(fig)
