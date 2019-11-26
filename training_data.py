import coinbase as cb
import pandas as pd
from datetime import datetime, timedelta

if __name__ == "__main__":
    end = datetime.utcnow() - timedelta(hours=1)
    cb.get_trades_by_range(cb.Coin.Bitcoin, cb.Granularity.Hour, end_datetime=end, skip_pages=1000)