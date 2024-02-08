# Classifiers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.ensemble import RandomForestClassifier # Random Forests
from sklearn.tree import DecisionTreeClassifier # C5.0 (Decision Tree)
from sklearn.neural_network import MLPClassifier # Neural Network
from sklearn.ensemble import GradientBoostingClassifier # Gradient Boosting Machine (GBM)
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
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

X = dataset[feature_names]
y = dataset[["SCS"]].to_numpy()

def clean_data(dataset):
    X = dataset[feature_names]
    for t in transformations:
        X = X.replace({t: transformations[t]})
    return X.to_numpy()


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

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = [
    "PRGS",
    "HUM_1_FW",
    "HUM_1_AGE",
    "HUM_1_STA",
    "HUM_2_FW",
    "HUM_2_AGE",
    "HUM_2_STA"
]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50)),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

print("Building classifiers...")

# Logistic Regression
lr_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(random_state=SEED))])
#lr_model = LogisticRegression(random_state=SEED)
lr_model.fit(X_train, y_train)
lr_model_AUC = round(roc_auc_score(y_test, lr_model.predict_proba(X_test)[:,1]), 3)

# Random Forests
rf_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=SEED, n_jobs = 10))])
#rf_model = RandomForestClassifier(random_state=SEED, n_jobs = 10)
rf_model.fit(X_train, np.ravel(y_train))
#dump(rf_model, "{}ecsa2023_clf.joblib".format(DIR))
rf_model_AUC = round(roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]), 3)

# C5.0 (Decision Tree)
dt_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier(random_state=SEED))])
#dt_model = DecisionTreeClassifier(random_state=SEED)
dt_model.fit(X_train, y_train)
dt_model_AUC = round(roc_auc_score(y_test, dt_model.predict_proba(X_test)[:,1]), 3)

# Neural Network
nn_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", MLPClassifier(random_state=SEED))])
#nn_model = MLPClassifier(random_state=SEED)
nn_model.fit(X_train, np.ravel(y_train))
nn_model_AUC = round(roc_auc_score(y_test, nn_model.predict_proba(X_test)[:,1]), 3)

# Gradient Boosting Machine (GBM)
gbm_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", GradientBoostingClassifier(random_state=SEED))])
gbm_model2 = GradientBoostingClassifier(random_state=SEED)
gbm_model.fit(X_train, np.ravel(y_train))
gbm_model2.fit(clean_data(X_train), np.ravel(y_train))
gbm_model_AUC = round(roc_auc_score(y_test, gbm_model.predict_proba(X_test)[:,1]), 3)

# eXtreme Gradient Boosting Tree (xGBTree)
xgb_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", GradientBoostingClassifier(random_state=SEED))])
#xgb_model = xgb.XGBClassifier(random_state=SEED)
xgb_model.fit(X_train, y_train)
xgb_model_AUC = round(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]), 3)

# Summarise into a DataFrame
model_performance_df = pd.DataFrame(data=np.array([['Logistic Regression', 'Random Forests', 'C5.0 (Decision Tree)', 'Neural Network', 'Gradient Boosting Machine (GBM)', 'eXtreme Gradient Boosting Tree (xGBTree)'],
                            [lr_model_AUC, rf_model_AUC, dt_model_AUC, nn_model_AUC, gbm_model_AUC, xgb_model_AUC]]).transpose(),
             index = range(6),
             columns = ['Model', 'AUC'])
model_performance_df['AUC'] = model_performance_df.AUC.astype(float)
model_performance_df = model_performance_df.sort_values(by = ['AUC'], ascending = False)

# Visualise the performance of models
#print(model_performance_df)
#model_performance_df.plot(kind = 'barh', y = 'AUC', x = 'model', figsize=(6,3))
#plt.xlim(0, 1.0)
#plt.tight_layout()
#plt.show()
#plt.savefig('model_auc.pdf')

# generate the 10-fold Cross Validation AUC
print("Generating 10-fold Cross Validation AUC...")
my_roc_auc = make_scorer(roc_auc_score, multi_class='ovo', needs_proba=True)
#X_full = np.concatenate((X_train, X_test))
#y_full = np.concatenate((y_train, y_test))

model_performance_df = pd.DataFrame()
#model_performance_df['LR'] = cross_val_score(lr_model, X, y.ravel(), cv = CV_KFOLD, scoring = my_roc_auc)
model_performance_df['RF'] = cross_val_score(rf_model, X, y.ravel(), cv = CV_KFOLD, scoring = my_roc_auc)
model_performance_df['DT'] = cross_val_score(dt_model, X, y.ravel(), cv = CV_KFOLD, scoring = my_roc_auc)
model_performance_df['NN'] = cross_val_score(nn_model, X, y.ravel(), cv = CV_KFOLD, scoring = my_roc_auc)
model_performance_df['GBM'] = cross_val_score(gbm_model, X, y.ravel(), cv = CV_KFOLD, scoring = my_roc_auc)
model_performance_df['XGB'] = cross_val_score(xgb_model, X, y.ravel(), cv = CV_KFOLD, scoring = my_roc_auc)

print("Exporting...")
# export to csv, display, and visualise the data frame
model_performance_df.to_csv("{}auc_dataset{}.csv".format(DIR, POINTS), index = False)

#matplotlib.rcParams.update({'font.size': 12})

#print("Plotting...")
#ax = model_performance_df.plot.box(ylim = (0.3, 1.0), ylabel = 'AUC', showmeans=True)
#ax.set_aspect(5.0)
#plt.show()
#plt.savefig('plots/xvalidation_auc.pdf')

print("Done.")


