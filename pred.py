import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import matplotlib.pyplot as plt
import sklearn.ensemble
from sklearn import datasets
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
import seaborn as sns
import math

def linear_regression(X, Y):
    lr = LinearRegression(normalize=True)
    lr.fit(X, Y)
    return lr
def xgboosting(X, Y):
    xg_reg = xgb.XGBRegressor(normalize=True)
    xg_reg.fit(X,Y)
    return xg_reg

def correlation_tests(df):
    corr = df.corr()
    corr.to_html('graphs/correlation.html')

    heatmap = sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns)
    heatmap.figure.savefig("graphs/heatmap.png", bbox_inches = "tight")
    plt.close()

def split_dataset(data, test_percent: int):
    size = len(data)-1
    test_size = math.ceil(size * (test_percent/100))
    return data[:test_size].copy(), data[test_size:].copy()

def models_and_test_set(df):
    df = df.drop(0)
    df = df.dropna()
    df = df.drop(columns=['time', 'Day','open','z_score','low','high','price_mean'])
    test, train = split_dataset(df, 20)
    X_train, Y_train = train.drop(columns=['output']), train['output']
    return linear_regression(X_train, Y_train), xgboosting(X_train, Y_train), test

def rmse(y_pred, y_true):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def percent_over_under_true(df_true_vs_pred: pd.DataFrame, pred_over=True):
    if pred_over:
        return len(df_true_vs_pred[df_true_vs_pred['true'] < df_true_vs_pred['pred']])/len(df_true_vs_pred)
    else:
        return len(df_true_vs_pred[df_true_vs_pred['true'] > df_true_vs_pred['pred']])/len(df_true_vs_pred)

type_hourly ='csv/processed/BTC_with_news.csv'
type_daily = 'csv/processed/BTC_with_news_daily.csv'

df_hourly = pd.read_csv(type_hourly)
df_hourly['output'] = df_hourly['high'].shift(1)

df_daily = pd.read_csv(type_daily)
df_daily['output'] = df_daily['high'].shift(1)

lr_hour, xgbst_hour, test_hour = models_and_test_set(df_hourly)
lr_day, xgbst_day, test_day = models_and_test_set(df_daily)

X_test_hour, Y_test_hour = test_hour.drop(columns=['output']), test_hour['output']
X_test_day, Y_test_day = test_day.drop(columns=['output']), test_day['output']

lr_pred_daily, lr_pred_hourly = lr_day.predict(X_test_day), lr_hour.predict(X_test_hour)
xgb_pred_daily, xgb_pred_hourly = xgbst_day.predict(X_test_day), xgbst_hour.predict(X_test_hour)

# Daily
lr_daily_pred_vs_true = pd.DataFrame()
lr_daily_pred_vs_true['pred'] = lr_pred_daily
lr_daily_pred_vs_true['true'] = Y_test_day.to_numpy()

xgbst_daily_pred_vs_true = pd.DataFrame()
xgbst_daily_pred_vs_true['pred'] = xgb_pred_daily
xgbst_daily_pred_vs_true['true'] = Y_test_day.to_numpy()

# Hourly
lr_hourly_pred_vs_true = pd.DataFrame()
lr_hourly_pred_vs_true['pred'] = lr_pred_hourly
lr_hourly_pred_vs_true['true'] = Y_test_hour.to_numpy()

xgbst_hourly_pred_vs_true = pd.DataFrame()
xgbst_hourly_pred_vs_true['pred'] = xgb_pred_hourly
xgbst_hourly_pred_vs_true['true'] = Y_test_hour.to_numpy()

# RMSE
df_rmse = pd.DataFrame()
df_rmse['lr_hour'] = [rmse(lr_hourly_pred_vs_true['pred'], lr_hourly_pred_vs_true['true'])]
df_rmse['lr_daily'] = [rmse(lr_daily_pred_vs_true['pred'], lr_daily_pred_vs_true['true'])]

df_rmse['xgb_hour'] = [rmse(xgbst_hourly_pred_vs_true['pred'], xgbst_hourly_pred_vs_true['true'])]
df_rmse['xgb_daily'] = [rmse(xgbst_daily_pred_vs_true['pred'], xgbst_daily_pred_vs_true['true'])]

print("RMSE")
print(df_rmse)

# Over
df_over = pd.DataFrame()
df_over['lr_over_hour'] = [percent_over_under_true(lr_hourly_pred_vs_true)]
df_over['lr_over_daily'] = [percent_over_under_true(lr_daily_pred_vs_true)]

df_over['xgb_over_hour'] = [percent_over_under_true(xgbst_hourly_pred_vs_true)]
df_over['xgb_over_daily'] = [percent_over_under_true(xgbst_daily_pred_vs_true)]

print("Percent Over")
print(df_over)

# Under
df_under = pd.DataFrame()
df_under['lr_under_hour'] = [percent_over_under_true(lr_hourly_pred_vs_true, False)]
df_under['lr_under_daily'] = [percent_over_under_true(lr_daily_pred_vs_true, False)]

df_under['xgb_under_hour'] = [percent_over_under_true(xgbst_hourly_pred_vs_true, False)]
df_under['xgb_under_daily'] = [percent_over_under_true(xgbst_daily_pred_vs_true, False)]

print("Percent Under")
print(df_under)
