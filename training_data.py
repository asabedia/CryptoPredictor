import coinbase as cb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import iso8601 as iso
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AffinityPropagation

def get_missing_time_ranges():
    df = pd.read_csv("csv/processed/processed_BTC.csv")
    t_df = df['time'].apply(lambda time: str(iso.parse_date(time).hour))
    print(t_df)
    value = int(t_df[0])
    print(value)
    first = True
    missing = []
    for index, row in t_df.iteritems():

        def get_expected_value(current_value):
            if current_value == 0:
                return 23
            else: 
                return current_value - 1

        if (not first and get_expected_value(value) != int(row)):
            missing.append({"index": index, "expected_hour": get_expected_value(value), "previous_time": df['time'].iloc[index-1], "current_time": df['time'].iloc[index], "previous_trade_id": int(df['trade_id'].iloc[index - 1])})
            value = int(row)
        else:
            value = int(row)
            if first:
                first = False

    m_df = pd.DataFrame(missing)
    m_df.to_csv('csv/cleaning/missing_trade_intervals.csv', index=False)

def save_processed_data():
    df_trades = pd.read_csv("csv/trades_data_BTC.csv")

    df_hist = pd.read_csv("csv/historical_data_BTC.csv")

    df_trades['time'] = df_trades['time'].apply(lambda time: iso.parse_date(time).replace(minute=0, second=0, microsecond=0).isoformat())
    gb = df_trades.groupby('time')
    means = gb['price'].mean()
    stds = gb['price'].std()
    last_trade = gb['trade_id'].last()

    df_stats = pd.merge(means, stds, on='time', suffixes=('_mean', '_stdev'))
    df_stats = pd.merge(last_trade, df_stats, on='time')
    df_joined = pd.merge(df_hist, df_stats, how = 'left', on='time')
    df_joined['z_score'] = (df_joined['close']-df_joined['price_mean'])/df_joined['price_stdev']

    print(df_joined['z_score'].isna().sum())
    df_joined = df_joined.dropna()
    df_joined.to_csv('csv/processed/processed_BTC.csv', index=False)

def fill_missing(m_df, starting=1, skip=10):
    for i in range(starting, len(m_df)):
        cb.get_trades_by_range(cb.Coin.Bitcoin, end_datetime=iso.parse_date(m_df.loc[i, 'current_time']), skip_pages=skip, append_existing=True, after=m_df.loc[i, 'previous_trade_id'])

def clean(df=pd.read_csv('csv/processed/processed_BTC.csv')):
    before = len(df)
    df = df[df.price_stdev != 0]
    after = len(df)
    if 'trade_id' in df.columns:
        df = df.drop(columns=['trade_id'])
    print("removed %s rows" % (before - after))
    df.to_csv('csv/processed/processed_cleaned_BTC.csv', index=False)
    return df

if __name__ == "__main__":
    # 2018-12-31T16:00:00
    # end = datetime(year=2018, day=31, month=12, tzinfo=timezone.utc)
    # end = datetime.utcnow() - timedelta(hours=1)
    # cb.get_trades_by_range(cb.Coin.Ethereum, end_datetime=end)
    # cb.get_historical_df_by_end_datetime(cb.Coin.Bitcoin, end_datetime=end, granularity=cb.Granularity.Hour)
    # df = pd.read_csv("csv/processed/processed_cleaned_BTC.csv")
    pass
    
    
