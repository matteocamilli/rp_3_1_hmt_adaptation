# Classifiers
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.ensemble import RandomForestRegressor # Random Forests
from sklearn.tree import DecisionTreeRegressor # C5.0 (Decision Tree)
from sklearn.neural_network import MLPRegressor # Neural Network
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor # Gradient Boosting Machine (GBM)
import xgboost as xgb # eXtreme Gradient Boosting Tree (xGBTree)
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
# AUC calculation
from sklearn.metrics import roc_auc_score
# Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectPercentile, chi2

POINTS = 1000
SUBSET = 1000
SEED = 1234
CV_KFOLD = 10

DIR = "data/ecsa2023ext/"

# PRGS,ORCH_1_Dstop,ORCH_1_Drestart,ORCH_1_Fstop,ORCH_1_Frestart,PSCS__TAU,HUM_1_VEL,HUM_2_VEL,HUM_1_FW,HUM_1_AGE,HUM_1_STA,HUM_2_FW,HUM_2_AGE,HUM_2_STA,HUM_1_POS_X,HUM_1_POS_Y,HUM_2_POS_X,HUM_2_POS_Y,ROB_1_VEL,ROB_1_CHG,PRSCS_LB,PRSCS_UB,FTG_HUM_1_LB,FTG_HUM_1_UB,FTG_HUM_2_LB,FTG_HUM_2_UB

dataset = pd.read_csv("{}dataset{}.csv".format(DIR, POINTS)).head(SUBSET)
feature_names = [
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

numeric_features = [
    "ORCH_1_Dstop",
    "ORCH_1_Drestart",
    "ORCH_1_Fstop",
    "ORCH_1_Frestart",
    "PSCS__TAU",
    "HUM_1_VEL",
    "HUM_2_VEL",
    "HUM_1_POS_X",
    "HUM_1_POS_Y",
    "HUM_2_POS_X",
    "HUM_2_POS_Y",
    "ROB_1_VEL",
    "ROB_1_CHG"
]

categorical_features = [
    "PRGS",
    "HUM_1_FW",
    "HUM_1_AGE",
    "HUM_1_STA",
    "HUM_2_FW",
    "HUM_2_AGE",
    "HUM_2_STA"
]

def clean(dataset):
    X = dataset[feature_names]
    for t in transformations:
        X = X.replace({t: transformations[t]})
    return X


X = clean(dataset)
y = dataset[["FTG_HUM_1"]].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_train.to_csv("regressors/X_train_FTG", index=False)

print("Building Regressors...")

# Random Forests
rf_model = RandomForestRegressor(random_state=SEED, n_jobs = 1)
rf_model.fit(X_train, np.ravel(y_train))
dump(rf_model, "regressors/regressor_FTG_LIME.joblib")
