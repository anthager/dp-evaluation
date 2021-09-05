import time
from calendar import timegm
from datetime import datetime, timezone

def timestamp_to_epoch(ts):
  utc_time = time.strptime(ts, "%Y-%m-%d %H:%M:%S")
  epoch_time = timegm(utc_time)
  return epoch_time

def epoch_to_timestamp(epoch):
  return datetime.fromtimestamp(epoch, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

