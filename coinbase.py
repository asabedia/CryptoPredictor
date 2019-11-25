import requests as req
from datetime import datetime, timedelta
from enum import Enum, auto
import csv
import time
import pandas as pd

_base_url = "https://api.pro.coinbase.com/%s"
_csv_path = "csv/%s"
_rate_limit = 3 # requests per second

class Coin(Enum):
        Bitcoin = "BTC"
        Litecoin = "LTC"
        Ethereum = "ETH"

class Granularity(Enum):
    Hour = {'time_unit': 'hours' , 'seconds_per_unit_time': 3600}

def get_time_frame(granularity, count: int, start=datetime.now()) -> (datetime, datetime):
    max_count_for_frame = 300
    if count > max_count_for_frame:
        count = max_count_for_frame
    delta = {granularity.value['time_unit']: count}
    end = start
    start = start - timedelta(**delta)
    return (start, end)

def get_historical_df_by_count(coin, count=5, currency_code="USD", granularity=Granularity.Hour, **kwrgs):
    csv_name = "historical_data_%s.csv" % coin.value
    headers = ['time', 'low', 'high', 'open', 'close', 'volume']
    historical_data = "products/%s/candles" % (coin.value + "-" + currency_code)
    
    def process_results(results: pd.DataFrame, first: bool):
        csv = _csv_path % csv_name
        results['time'] = results['time'].apply(lambda x: datetime.fromtimestamp(x).isoformat())
        results = results.drop(columns='volume')
        if first:
            results.to_csv(csv, index=False)
        else: 
            results.to_csv(csv, mode='a', header=False, index=False)

    first = True
    total_requests = 0

    remaining_count = count
    end = datetime.now()
    start = end
    while(remaining_count > 0):
        (start, end) = get_time_frame(granularity=granularity, count=remaining_count, start=start)
        
        params = {
            'start': start.isoformat(), 
            'end': end.isoformat(),
            'granularity': granularity.value['seconds_per_unit_time']
        }
        url = _base_url % historical_data

        # wait for rate limit to relax
        if (total_requests % _rate_limit == 0):
            time.sleep(3)

        resp = req.get(url, params=params)
        total_requests = total_requests + 1
        if resp.status_code == 200:
            resp_conent=resp.json()

            if (not resp_conent):
                break
            else:
                remaining_count = remaining_count - len(resp_conent)

            df = pd.DataFrame(resp_conent)
            df.columns = headers
            process_results(df, first)

            if first:
                first = False

        else: 
            return "Error for %i'th call, start: %s end: %s" % (total_requests, start.isoformat(), end.isoformat())

    time.sleep(1)
    return "Made %i calls processed. final start: %s final end: %s" % (total_requests, start.isoformat(), end.isoformat())