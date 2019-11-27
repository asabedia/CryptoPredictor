import coinbase as cb
import pandas as pd
from datetime import datetime, timedelta, timezone

if __name__ == "__main__":
    # 2018-12-31T16:00:00
    end = datetime(year=2018, day=31, month=12, tzinfo=timezone.utc)
    # end = datetime.utcnow() - timedelta(hours=1)
    cb.get_trades_by_range(cb.Coin.Bitcoin, end_datetime=end, skip_pages=3000)