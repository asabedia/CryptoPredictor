import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import matplotlib.pyplot as plt
import sklearn.ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
import xgboost as xgb
from sklearn import preprocessing
import math

def xgb_kfold(X, Y, splits):
    data_dmatrix = xgb.DMatrix(data=X,label=Y)
    params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=splits,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    return (cv_results["test-rmse-mean"]).tail(1)

# XGBoost
def xgboosting(X, Y):
    xg_reg = xgb.XGBRegressor(n_estimators = 1000)
    xg_reg.fit(X,Y)
    return xg_reg

# Linear Regression
def linear_regression(X, Y):
    lr = LinearRegression(normalize=True)
    lr.fit(X, Y)
    return lr

def split_dataset(data, test_percent: int):
    size = len(data)-1
    test_size = math.ceil(size * (test_percent/100))
    return data[:test_size].copy(), data[test_size:].copy()

df = pd.read_csv('csv/processed/BTC_with_news.csv')
df['output'] = df['close'].shift(1)
df = df.drop(0)
df = df.dropna()

df = df.drop(columns=['time', 'Day'])
test, train = split_dataset(df, 20)
print(test)
X_train = train.drop(columns=['output'])
X_test = test.drop(columns=['output'])
print(X_train)
Y_train = train['output']
Y_test = test['output']
print(X_train)
xgbm = xgboosting(X_train, Y_train)
lr = linear_regression(X_train, Y_train)
pdf = pd.DataFrame(columns=['pred', 'true'])
pdf['pred'] = xgbm.predict(X_test)
pdf['true'] = Y_test.to_numpy()
print("XGB RMSE: ", np.sqrt(mean_squared_error(Y_test, xgbm.predict(X_test))))
print("LinReg RMSE: ", np.sqrt(mean_squared_error(Y_test, lr.predict(X_test))))
print(pdf)
# xgb.plot_importance(xgbm, height=0.9)