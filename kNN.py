import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

df = pd.read_excel('FHK2022_dataA.xlsx')

col_list = [col for col in df.columns]
num_nan_list = [df[col].isna().sum() for col in col_list]

# Remove NAN and some columns
remove_labels = df.drop(labels=['Work Time', 'Serial No.'], axis=1)
null_cols = remove_labels.columns[remove_labels.isnull().any()]
no_nan_df = remove_labels.drop(null_cols, axis=1)
copy = no_nan_df

# Normalization
part_column = no_nan_df.loc(axis=1)['Part Name']
no_nan_df = no_nan_df.drop(labels=['Part Name'], axis=1)
for col in no_nan_df.columns:
    if no_nan_df[col].std():
        no_nan_df[col] = (no_nan_df[col] - no_nan_df[col].mean()) / no_nan_df[col].std()
    else:
        no_nan_df[col] = 0
no_nan_df = pd.concat([part_column, no_nan_df], axis=1)

# Train, Validation set
x = no_nan_df.drop(labels=['G', 'T1', 'T2', 'W_R', 'W_L'], axis=1)
y = copy.loc(axis=1)[['Part Name', 'G', 'T1', 'T2', 'W_R', 'W_L']]
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=True)

# Divide by Parts (A, B, C)
partA_x_train = x_train[x_train['Part Name'] == 'A'].drop(labels='Part Name', axis=1)
partA_x_valid = x_valid[x_valid['Part Name'] == 'A'].drop(labels='Part Name', axis=1)
partA_y_train = y_train[y_train['Part Name'] == 'A'].drop(labels='Part Name', axis=1)
partA_y_valid = y_valid[y_valid['Part Name'] == 'A'].drop(labels='Part Name', axis=1)

# As a whole
x_train = x_train.drop(labels='Part Name', axis=1)
x_valid = x_valid.drop(labels='Part Name', axis=1)
y_train = y_train.drop(labels='Part Name', axis=1)
y_valid = y_valid.drop(labels='Part Name', axis=1)

# KNN without PCA (65 input params -> 5 output params)
num_neighbor = 4
clf = KNeighborsRegressor(n_neighbors=num_neighbor, weights='distance')
clf.fit(partA_x_train, partA_y_train)
prediction = clf.predict(partA_x_valid)
print((partA_y_valid - prediction).mean())

clf2 = KNeighborsRegressor(n_neighbors=num_neighbor, weights='distance')
clf.fit(x_train, y_train)
prediction = clf.predict(x_valid)
print((y_valid - prediction).mean())
