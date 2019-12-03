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
from pred import split_dataset, linear_regression, xgboosting, correlation_tests

type_daily = 'csv/processed/BTC_with_news_daily.csv'
df = pd.read_csv(type_daily)

# GRAPH ENTIRE DATASET ##
df_rev = df.iloc[::-1]
figure = plt.figure(figsize = (6,6))
plt.plot(df_rev['Day'], df_rev['price_mean'])
plt.title('BTC Price Over Time ')
figure.savefig('graphs/BTC_Price_Over_Time.png',bbox_inches = "tight")
plt.close()

## POST PROCESSING ##
df['output'] = df['high'].shift(1)
df = df.drop(0)
df = df.dropna()
df = df.drop(columns=['time', 'Day','open','z_score','low','high','price_mean'])


## CORRELATION TESTS ##
df_corr = df.drop(columns=['output'])
correlation_tests(df_corr)

## MODEL TRAINING ##
test, train = split_dataset(df, 20)
X_train = train.drop(columns=['output'])
X_test = test.drop(columns=['output'])
Y_train = train['output']
Y_test = test['output']
xgbm = xgboosting(X_train, Y_train)
lr = linear_regression(X_train, Y_train)

## OUTPUT ##

#LinReg
pdf = pd.DataFrame(columns=['true'])
pdf['pred_Lin_Reg'] = lr.predict(X_test)
pdf['true'] = Y_test.to_numpy()
print("LinReg RMSE: ", np.sqrt(mean_squared_error(Y_test, lr.predict(X_test))))
print(pdf)

#Xgboost
pdf2 = pd.DataFrame(columns=['pred', 'true'])
pdf2['pred'] = xgbm.predict(X_test)
pdf['pred_XGBoost'] = xgbm.predict(X_test)
pdf2['true'] = Y_test.to_numpy()
print("XGB RMSE: ", np.sqrt(mean_squared_error(Y_test, xgbm.predict(X_test))))
print(pdf2)

## FEATURE IMPORTANCE Linear Regression##
f_imp_lr = df.corr()
print(f_imp_lr['output'])

## FEATURE IMPORTANCE XGBoost##
feat_imp = pd.Series(xgbm.feature_importances_,index=X_train.transpose().index).sort_values(ascending=False)
feature_importance = plt.figure(figsize = (6,6))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("XGB Feature Importance")
plt.legend()
feat_imp.to_frame().rename(columns={0: "Feature Importance"}).to_html('graphs/Feature_Importance.html')
feature_importance.savefig("graphs/XGB_Feature_Importance.png", bbox_inches = "tight")
plt.close()

## Predictions ##
test, train = split_dataset(df2, 20)
pdf['time'] = test['time']
figure = plt.figure(figsize = (12,6))
pdf = pdf.iloc[::-1]
plt.plot( 'time', 'pred_Lin_Reg', data=pdf, marker='',  color='red', linewidth=1, label='Linear Regression')
plt.plot( 'time', 'pred_XGBoost', data=pdf, marker='', color='green', linewidth=1,label="XGBoost")
plt.plot( 'time', 'true', data=pdf, marker='', color='black', linewidth=2,label="Actual",alpha=0.5)
plt.xticks(pdf['time'][::336], rotation='vertical')
plt.title('Hourly Predictions')
plt.legend()
figure.savefig("graphs/predictions.png", bbox_inches = "tight")

plt.close()