import pandas as pd
import numpy as np
from preprocess import preprocess
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

x, y, x1, x2, x3 = preprocess(output_param='G')
x, y = x.to_numpy(), y.to_numpy().reshape(-1, 1)
x_train, x_valid_test, y_train, y_valid_test = train_test_split(x, y, test_size=0.2, shuffle=True)
x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test, test_size=0.5, shuffle=True)

# regressor = XGBRegressor(n_estimators=500, max_depth=9, min_child_weight=1.2, colsample_bytree=1.0)
# regressor.fit(x_train, y_train, verbose=False)

# parameters = {"colsample_bytree":[0.8, 1.0],"min_child_weight":[1.0, 1.2],
#               'max_depth': [6, 7, 8, 9], 'n_estimators': [500, 1000]}
# regression_cv = GridSearchCV(regressor, parameters, scoring='neg_mean_absolute_error', verbose=10)
# regression_cv.fit(x_train, y_train)
# best_params = regression_cv.best_params_

best_params = {'colsample_bytree': 1.0, 'max_depth': 6, 'min_child_weight': 1.2, 'n_estimators': 500}

regressor = XGBRegressor(n_estimators=500, max_depth=6, min_child_weight=1.2, colsample_bytree=1.0)
regressor.fit(x_train, y_train, verbose=False)
prediction = regressor.predict(x_test)

print(prediction)
print(np.sum(np.abs(prediction-y_test)))
print(np.mean(np.abs(prediction-y_test)))

