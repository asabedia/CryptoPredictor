import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import sklearn.ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

df = pd.read_csv('csv/processed/BTC_with_news.csv')
df['output'] = df['close'].shift(1)
df = df.drop(columns=['time','Day'])
df = df.drop(0)
df = df.dropna()

def gradient_boosting():
    df_new = pd.DataFrame(columns =['predicted','actual'])
    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['output']), df['output'], test_size=0.2,random_state=42)
    xgb = sklearn.ensemble.GradientBoostingRegressor()
    xgb.fit(x_train, y_train)
    # print("Gradient Boosting Accuracy: ", xgb.score(x_test, y_test))
    df_new['predicted'] = xgb.predict(x_test)
    df_new['actual'] = y_test.values
    # print(y_test.values)
    print(df_new)





gradient_boosting()

# print(df)