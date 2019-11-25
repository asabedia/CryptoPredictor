import coinbase as cb
import pandas as pd
from datetime import datetime

if __name__ == "__main__":
    cb.get_historical_df_by_end_datetime(cb.Coin.Ethereum, datetime(2019, 1, 1), cb.Granularity.Hour)
    cb.get_historical_df_by_end_datetime(cb.Coin.Litecoin, datetime(2019, 1, 1), cb.Granularity.Hour)
