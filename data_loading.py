import pandas as pd

from scipy.stats import chi2_contingency, chi2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, tree

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import  mean_squared_error, mean_absolute_error


def load_train(data):
    X = pd.read_csv(data, index_col = 0)
    return X

def load_test(data):
    y = pd.read_csv(data, index_col = 0)
    return y

def extratree_regressor(X, y):
    #Train test plit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 123)
    #Extratree Regressor
    etr = ExtraTreesRegressor(max_depth= 60, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 150)
    etr.fit(X_train, y_train)
    #predict y value from X_test
    y_train_pred = etr.predict(X_train)
    y_test_pred = etr.predict(X_test)
    #Model evaluation
    #score original data (R^2)
    R_ori = etr.score(X, y)
    #score train
    R_train = etr.score(X_train, y_train)
    #score test
    R_test = etr.score(X_test, y_test)

    #mse original
    mse_ori = mean_squared_error(y, etr.predict(X))
    #mse train
    mse_train = mean_squared_error(y_train, y_train_pred)
    #mse test
    mse_test = mean_squared_error(y_test, y_test_pred)

    #mae original
    mae_ori = mean_absolute_error(y, etr.predict(X))
    #mae train
    mae_train = mean_absolute_error(y_train, y_train_pred)
    #mae test
    mae_test = mean_absolute_error(y_test, y_test_pred)

    #create dataframe
    col = {"R Squared": [R_ori, R_train, R_test],
            "Mean Squared Error": [mse_ori, mse_train, mse_test],
            "Mean Absolute Error":[mae_ori, mae_train, mae_test]}
    result = pd.DataFrame(d)
    result.index = ["Original", "Train", "Test"]
    return result
