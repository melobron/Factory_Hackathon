import optuna
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error


class AutoML:
    def __init__(self, data, target, regressor=XGBRegressor):
        # self.x_t, self.x_v, self.x_t = data[0], data[1], data[2]
        # self.y_t, self.y_v, self.y_t = target[0], target[1], target[2]
        self.x_train, self.x_test = data[0], data[1]
        self.y_train, self.y_test = target[0], target[1]
        self.regressor = regressor

    def objective(self, trial):
        if self.regressor == XGBRegressor:
            param = {
                'tree_method': 'gpu_hist',
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
                'learning_rate': trial.suggest_categorical('learning_rate',
                                                           [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
                'n_estimators': 4000,
                'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
                'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            }

        elif self.regressor == CatBoostRegressor:
            param = {
                'iterations': trial.suggest_int("iterations", 4000, 25000),
                'od_wait': trial.suggest_int('od_wait', 500, 2300),
                'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 1),
                'reg_lambda': trial.suggest_uniform('reg_lambda', 1e-5, 100),
                # 'subsample': trial.suggest_uniform('subsample', 0, 1),
                'random_strength': trial.suggest_uniform('random_strength', 10, 50),
                'depth': trial.suggest_int('depth', 1, 15),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
                'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),
                'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            }
        elif self.regressor == LGBMRegressor:
            param = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }

        model = self.regressor(**param)
        # model.fit(self.x_t, self.y_t, eval_set=[(self.x_v, self.y_v)], early_stopping_rounds=100, verbose=False)
        model.fit(self.x_train, self.y_train, early_stopping_rounds=100, verbose=False)

        # prediction = model.predict(self.x_v)
        # l1_loss = mean_absolute_error(self.y_v, prediction)
        prediction = model.predict(self.x_test)
        l1_loss = mean_absolute_error(self.y_test, prediction)
        return l1_loss


def Optuna(data, target, regressor=XGBRegressor, n_trials=100):
    automl_model = AutoML(data, target, regressor)

    study = optuna.create_study(direction='minimize')
    study.optimize(automl_model.objective, n_trials=n_trials)
    best_param = study.best_trial.params
    print('Best trial:', best_param)

    return best_param
