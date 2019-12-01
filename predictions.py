import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import sklearn.ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
import xgboost as xgb
from sklearn import preprocessing

def xgb_kfold(X, Y, splits):
    data_dmatrix = xgb.DMatrix(data=X,label=Y)
    params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=splits,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    return (cv_results["test-rmse-mean"]).tail(1)


# XGBoost
def xgboosting(X, Y):
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.5,
                max_depth = 10, alpha = 10, n_estimators = 20)
    xg_reg.fit(X,Y)
    return xg_reg

# Linear Regression
def kfold_linear_regression(x_train, x_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    y_true = y_test
    lr.score
    return {
        'mse': mean_squared_error(y_pred, y_true),
        'score': r2_score(y_pred, y_true)
    } 

# Linear Regression
def linear_regression(X, Y):
    lr = LinearRegression()
    lr.fit(X, Y)
    return lr

# KFold
def kfold_validation(splits, features, target, model):
    kf = KFold(n_splits=splits)
    features = features.to_numpy()
    target = target.to_numpy()
    k_sum = {
        'rmse': []
    }
    for train_index, test_index in kf.split(features):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]
        regression = model(x_train, x_test, y_train, y_test)
        k_sum['rmse'].append(np.sqrt(regression['mse']))
    
    print("K: ", splits)
    print("rmse: ", np.mean(k_sum['rmse']))
    print("min_mse: ", np.min(k_sum['rmse']))
    print("max_mse: ", np.max(k_sum['rmse']))

df = pd.read_csv('csv/processed/BTC_with_news.csv')
df['output'] = df['close'].shift(1)
df = df.drop(columns=['time','Day','price_mean'])
df = df.drop(0)
df = df.dropna()
X = preprocessing.scale(df.drop(columns=['output']))
Y = df['output']

# joing = pd.DataFrame()
# test =  X[250:350]
# test_true = Y[250:350]
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=500)

# # print(xgboosting(X_train, X_test, y_train, y_test))
# lr = boosting(X, Y.to_numpy())
# joing['pred'] = lr.predict(test)
# joing['true'] = test_true.to_numpy()
# print(mean_squared_error(joing['pred'], joing['true']))
# # join = np.concatenate(test, test_true)
# print(joing)

print(xgb_kfold(X, Y, 3))
kfold_validation(3, df.drop(columns=['output']), df['output'], kfold_linear_regression)

xgb = xgboosting(X, Y)
lr = linear_regression(X, Y)

print(xgb)
print(lr.coef_)
