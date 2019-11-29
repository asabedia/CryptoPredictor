import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import sklearn.ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('csv/processed/BTC_with_news.csv')
df['output'] = df['close'].shift(1)
df = df.drop(columns=['time','Day','price_mean'])
df = df.drop(0)
df = df.dropna()



def random_forest():
    df_new = pd.DataFrame(columns =['predicted','actual'])
    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['output']), df['output'], test_size=0.2,random_state = 30)
    rfr = sklearn.ensemble.RandomForestRegressor()
    rfr.fit(x_train, y_train)
    df_new['predicted'] = rfr.predict(x_test)
    df_new['actual'] = y_test.values
    print(mean_squared_error(df_new['actual'],df_new['predicted']))

def gradient_boosting():
    df_new = pd.DataFrame(columns =['predicted','actual'])
    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['output']), df['output'], test_size=0.2,random_state = 30)
    xgb = sklearn.ensemble.GradientBoostingRegressor()
    xgb.fit(x_train, y_train)
    df_new['predicted'] = xgb.predict(x_test)
    df_new['actual'] = y_test.values
    print(mean_squared_error(df_new['actual'],df_new['predicted']))

def linear_regression():
    df_new = pd.DataFrame(columns =['predicted','actual'])
    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['output']), df['output'], test_size=0.2,random_state = 30)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    df_new['predicted'] = lr.predict(x_test)
    df_new['actual'] = y_test.values
    print(mean_squared_error(df_new['actual'],df_new['predicted']))

# print(df.corr().abs())
#
# df.corr().abs().to_csv('csv/correlation.csv')

print("Gradient Boosting MSE:")
gradient_boosting()
print("Random Forest MSE:" )
random_forest()
print("Linear Regression MSE:" )
linear_regression()
# print(df)