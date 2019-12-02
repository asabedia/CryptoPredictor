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

type_hourly ='csv/processed/BTC_with_news.csv'
type_daily = 'csv/processed/BTC_with_news_daily.csv'
df = pd.read_csv(type_daily)

## GRAPH ENTIRE DATASET ##
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
df = df.drop(columns=['time', 'Day','open','z_score'])


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
pdf = pd.DataFrame(columns=['pred', 'true'])
pdf['pred'] = lr.predict(X_test)
pdf['true'] = Y_test.to_numpy()
print("LinReg RMSE: ", np.sqrt(mean_squared_error(Y_test, lr.predict(X_test))))
print(pdf)

#Xgboost
pdf2 = pd.DataFrame(columns=['pred', 'true'])
pdf2['pred'] = xgbm.predict(X_test)
pdf2['true'] = Y_test.to_numpy()
print("XGB RMSE: ", np.sqrt(mean_squared_error(Y_test, xgbm.predict(X_test))))
print (pdf2)

## FEATURE IMPORTANCE ##
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