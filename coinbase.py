import requests as req
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
import csv
import time
import math
import iso8601
import pandas as pd

_base_url = "https://api.pro.coinbase.com/%s"
_csv_path = "csv/%s"
_rate_limit = 3 # requests per second

class Coin(Enum):
        Bitcoin = "BTC"
        Litecoin = "LTC"
        Ethereum = "ETH"

class Granularity(Enum):
    Hour = {'time_unit': 'hours' , 'seconds_per_unit_time': 3600, 'unit_time_per_day': 24}

def get_csv_path_for(coin, prefix):
    fileName = "%s_%s.csv" % (prefix, coin.value)
    return _csv_path % fileName 

def get_time_frame(granularity, count: int, start=datetime.now()) -> (datetime, datetime):
    max_count_for_frame = 300
    if count > max_count_for_frame:
        count = max_count_for_frame
    delta = {granularity.value['time_unit']: count}
    end = start
    start = start - timedelta(**delta)
    return (start, end)

def get_historical_df_by_end_datetime(coin, end_datetime: datetime, granularity: Granularity, start_datetime=datetime.now()):
    delta = (start_datetime - end_datetime)
    hours = delta.days * granularity.value['unit_time_per_day'] + math.ceil(delta.seconds/granularity.value['seconds_per_unit_time'])
    get_historical_df_by_count(coin, count=hours, granularity=granularity)

def get_historical_df_by_count(coin, count=5, currency_code="USD", granularity=Granularity.Hour, **kwrgs):
    csv =  get_csv_path_for(coin, "historical_data")
    headers = ['time', 'low', 'high', 'open', 'close', 'volume']
    historical_data = "products/%s/candles" % (coin.value + "-" + currency_code)
    
    def process_results(results: pd.DataFrame, first: bool):
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
            print("Error for %i'th call, start: %s end: %s" % (total_requests, start.isoformat(), end.isoformat()))

    time.sleep(1)
    print ("Made %i calls processed. final start: %s final end: %s" % (total_requests, start.isoformat(), end.isoformat()))

def get_trades_by_range(coin: Coin, granularity: Granularity, end_datetime: datetime, skip_pages=10, currency_code="USD" ,start_datetime=datetime.now(timezone.utc)):
    csv = get_csv_path_for(coin, "trades_data")
    headers = ['time', 'trade_id', 'price', 'size', 'side']
    trades_data = "products/%s/trades" % (coin.value + "-" + currency_code)
    end_datetime = end_datetime.replace(tzinfo=timezone.utc)

    def reduce(date_time: datetime) -> datetime:
        if granularity==Granularity.Hour:
            return date_time.replace(minute=0, second=0, microsecond=0)
        else:
            return date_time

    def process_results(results: pd.DataFrame, first: bool):
        results['time'] = results['time'].apply(lambda x: reduce(iso8601.parse_date(x)).isoformat())
        if first:
            results.to_csv(csv, index=False)
        else: 
            results.to_csv(csv, mode='a', header=False, index=False)

    url = _base_url % trades_data

    last_datetime = start_datetime
    total_requests = 0
    first = True

    params = {}
    after = None
    print(end_datetime)
    while(last_datetime > end_datetime):
        if not first and after:
            print(str(int(after) - skip_pages))
            params['after'] = str(int(after) - skip_pages)

        # wait for rate limit to relax
        if (total_requests % _rate_limit == 0 and total_requests > 0):
            time.sleep(3)

        resp = req.get(url, params=params)
        total_requests = total_requests + 1
        if resp.status_code == 200:
            after = resp.headers['cb-after']
            resp_conent=resp.json()

            if (not resp_conent):
                break

            df = pd.DataFrame(resp_conent)
            df.columns = headers
            last_datetime = iso8601.parse_date(df['time'].iloc[-1])
            print(last_datetime)
            process_results(df, first)
            
            if first:
                first = False

        else: 
            print("Error for %i'th call, after: %s" % (total_requests, after))