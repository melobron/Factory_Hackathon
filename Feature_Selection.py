import numpy as np
import pandas as pd
import argparse
from preprocess import preprocess
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

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

# Preprocess data
if args.normalization:
    [x_train, x_valid, x_test], [y_train, y_valid, y_test], scaler = preprocess(args, output_param='G')
else:
    [x_train, x_valid, x_test], [y_train, y_valid, y_test] = preprocess(args, output_param='G')

# # Feature Selection
etc_model = ExtraTreesClassifier()
etc_model.fit(x_train, y_train)

print(etc_model.feature_importances_)
feature_list = pd.concat([])

