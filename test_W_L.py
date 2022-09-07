import numpy as np
import pandas as pd
import argparse
from preprocess import preprocess
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from Optuna import Optuna

# Arguments
parser = argparse.ArgumentParser(description='FHK2022')

# Outlier
parser.add_argument("--remove_out", type=bool, default=True, help="remove outliers")
parser.add_argument("--n_neighbors", type=int, default=30, help="remove outliers")
parser.add_argument("--metric", type=str, default='l2', help="distance metric")  # manhattan, l1, l2
parser.add_argument("--contamination", type=float, default=0.02, help="outlier percentage")

# Preprocessing
parser.add_argument("--normalization", type=bool, default=True, help="normalization")
parser.add_argument("--norm_type", type=str, default='Standard', help="remove outliers")  # MinMax, Standard

# Output Parameter
parser.add_argument("--output_param", type=str, default='G', help="output parameter")

args = parser.parse_args()
#######################################################################################################################

# Preprocess data
if args.normalization:
    # [x_train, x_valid, x_test], [y_train, y_valid, y_test], scaler = preprocess(args, output_param='G')
    [x_train, x_test], [y_train, y_test], scaler = preprocess(args, output_param=args.output_param)
else:
    # [x_train, x_valid, x_test], [y_train, y_valid, y_test] = preprocess(args, output_param='G')
    [x_train, x_test], [y_train, y_test] = preprocess(args, output_param=args.output_param)

# Input for TabNet
train_prediction, valid_prediction, test_prediction = [], [], []

# KNN
# regressor = KNeighborsRegressor(n_neighbors=5)
# regressor.fit(x_train, y_train)
# train_prediction.append(np.expand_dims(regressor.predict(x_train), axis=1))
# test_prediction.append(np.expand_dims(regressor.predict(x_test), axis=1))
#
# # RandomForest
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(x_train, y_train)
train_prediction.append(np.expand_dims(regressor.predict(x_train), axis=1))
test_prediction.append(np.expand_dims(regressor.predict(x_test), axis=1))
#
# # Decision Tree
regressor = DecisionTreeRegressor()
regressor.fit(x_train, y_train)
train_prediction.append(np.expand_dims(regressor.predict(x_train), axis=1))
test_prediction.append(np.expand_dims(regressor.predict(x_test), axis=1))

# AdaBoost
# regressor = AdaBoostRegressor(n_estimators=100)
# regressor.fit(x_train, y_train)
# train_prediction.append(np.expand_dims(regressor.predict(x_train), axis=1))
# test_prediction.append(np.expand_dims(regressor.predict(x_test), axis=1))
# #
# CatBoost
#### param tuning ####
# best_params = Optuna(data=(x_train, x_valid, x_test), target=(y_train, y_valid, y_test), regressor=CatBoostRegressor, n_trials=200)
# best_params = Optuna(data=(x_train, x_test), target=(y_train, y_test), regressor=CatBoostRegressor, n_trials=20)
# regressor = CatBoostRegressor(**best_params)
# regressor = CatBoostRegressor(**{'learning_rate': 0.007, 'depth': 12, 'l2_leaf_reg': 1.0, 'min_child_samples': 4})
#### param tuning ####
regressor = CatBoostRegressor()
regressor.fit(x_train, y_train)
train_prediction.append(np.expand_dims(regressor.predict(x_train), axis=1))
test_prediction.append(np.expand_dims(regressor.predict(x_test), axis=1))

# XGBoost
#### param tuning ####
# best_params = Optuna(data=(x_train, x_valid, x_test), target=(y_train, y_valid, y_test), regressor=XGBRegressor, n_trials=20)
# best_params = Optuna(data=(x_train, x_test), target=(y_train, y_test), regressor=XGBRegressor, n_trials=20)
# regressor = XGBRegressor(**best_params)
# regressor = XGBRegressor(**{'lambda': 0.06454133429711781, 'alpha': 0.4152564281978906, 'colsample_bytree': 0.8,
#                             'subsample': 0.7, 'learning_rate': 0.018, 'max_depth': 7, 'random_state': 24, 'min_child_weight': 4})
#### param tuning ####
regressor = XGBRegressor()
regressor.fit(x_train, y_train)
train_prediction.append(np.expand_dims(regressor.predict(x_train), axis=1))
test_prediction.append(np.expand_dims(regressor.predict(x_test), axis=1))

# LightGBM
#### param tuning ####
# best_params = Optuna(data=(x_train, x_valid, x_test), target=(y_train, y_valid, y_test), regressor=LGBMRegressor, n_trials=20)
# best_params = Optuna(data=(x_train, x_test), target=(y_train, y_test), regressor=LGBMRegressor, n_trials=20)
# regressor = LGBMRegressor(**best_params)
#### param tuning ####
regressor = LGBMRegressor()
regressor.fit(x_train, y_train)
train_prediction.append(np.expand_dims(regressor.predict(x_train), axis=1))
test_prediction.append(np.expand_dims(regressor.predict(x_test), axis=1))

# # TabNet
# regressor = TabNetRegressor()
# regressor.fit(X_train=x_train.to_numpy(), y_train=y_train.to_numpy().reshape(-1,1), eval_metric=['mae'],
#               max_epochs=1000, patience=300)
# train_prediction.append(regressor.predict(x_train.to_numpy()))
# test_prediction.append(regressor.predict(x_test.to_numpy()))

####### Stacking #######
# simple stacking
# e = np.zeros((test_prediction[0].shape[0], 1))
# for index in range(len(test_prediction)):
#     e += test_prediction[index] / len(test_prediction)
# test_prediction = e
# prediction = np.mean(test_prediction, axis=1).reshape(-1, 1)
# y_train, y_test = np.expand_dims(y_train, axis=1), np.expand_dims(y_test, axis=1)

# Other Stacking
train_prediction = np.concatenate(train_prediction, axis=1)
test_prediction = np.concatenate(test_prediction, axis=1)
y_train, y_test = np.expand_dims(y_train, axis=1), np.expand_dims(y_test, axis=1)

# # Stacking with TabNet
stacking = TabNetRegressor()
stacking.fit(X_train=train_prediction, y_train=y_train, eval_metric=['mae'],
              max_epochs=1000, patience=300)
prediction = stacking.predict(test_prediction)

# Stacking with Linear Regression
# stacking = LinearRegression()
# stacking.fit(train_prediction, y_train)
# prediction = stacking.predict(test_prediction)

# Unscale for Evaluation
if args.normalization:
    test_xy = np.concatenate((x_test, y_test), axis=1)
    prediction_xy = np.concatenate((x_test, prediction), axis=1)
    inv_xy = pd.DataFrame(scaler.inverse_transform(test_xy))
    pred_inv_xy = pd.DataFrame(scaler.inverse_transform(prediction_xy))
    y_test = inv_xy.iloc[:, -1]
    prediction = pred_inv_xy.iloc[:, -1]

print(np.sum(np.abs(prediction-y_test)))
print(np.mean(np.abs(prediction-y_test)))

