import pandas as pd
import numpy as np
from preprocess import preprocess
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import argparse

# Arguments
parser = argparse.ArgumentParser(description='FHK2022')

# Outlier
parser.add_argument("--remove_out", type=bool, default=False, help="remove outliers")
parser.add_argument("--n_neighbors", type=int, default=20, help="remove outliers")
parser.add_argument("--metric", type=str, default='l2', help="distance metric")  # manhattan, l1, l2
parser.add_argument("--contamination", type=float, default=0.02, help="outlier percentage")

# Preprocessing
parser.add_argument("--normalization", type=bool, default=False, help="normalization")
parser.add_argument("--norm_type", type=str, default='Standard', help="remove outliers")  # MinMax, Standard

args = parser.parse_args()
#######################################################################################################################

if args.normalization:
    [x_train, x_valid, x_test], [y_train, y_valid, y_test], scaler = preprocess(args, output_param='G')
else:
    [x_train, x_valid, x_test], [y_train, y_valid, y_test] = preprocess(args, output_param='G')

regressor = TabNetRegressor()
y_train, y_valid, y_test = np.expand_dims(y_train, axis=1), np.expand_dims(y_valid, axis=1), np.expand_dims(y_test, axis=1)
x_train, x_valid, x_test = x_train.to_numpy(), x_valid.to_numpy(), x_test.to_numpy()
regressor.fit(X_train=x_train, y_train=y_train, max_epochs=500, patience=300)
prediction = regressor.predict(x_test)

# Unscale for Evaluation
if args.normalization:
    test_xy = np.concatenate((x_test, y_test), axis=1)
    prediction_xy = np.concatenate((x_test, prediction), axis=1)
    inv_xy = pd.DataFrame(scaler.inverse_transform(test_xy))
    pred_inv_xy = pd.DataFrame(scaler.inverse_transform(prediction_xy))
    y_test = inv_xy.iloc[:, -1]
    prediction = pred_inv_xy.iloc[:, -1]

print(prediction)
print(np.sum(np.abs(prediction-y_test)))
print(np.mean(np.abs(prediction-y_test)))





