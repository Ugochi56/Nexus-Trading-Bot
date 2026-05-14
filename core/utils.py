from datetime import datetime
from core.config import *

def is_us_dst(date_obj):
    """Calculates if the given date is inside US Daylight Saving Time."""
    year = date_obj.year
    march_start = datetime(year, 3, 8)
    march_start = march_start.replace(day=14 - march_start.weekday()) 
    nov_end = datetime(year, 11, 1)
    nov_end = nov_end.replace(day=nov_end.day + (6 - nov_end.weekday()) % 7) 
    naive_date = date_obj.replace(tzinfo=None)
    return march_start <= naive_date < nov_end

def get_broker_hour_from_utc(utc_time):
    """
    Converts a raw UTC datetime into a static 'New York Aligned' Broker Hour.
    New York Close (5:00 PM NY) is universally anchored to Hour 0 (Midnight).
    """
    utc_hour = utc_time.hour
    if is_us_dst(utc_time):
        # Summer (EDT): NY Close is 21:00 UTC. To map 21 to 0, we add 3.
        return (utc_hour + 3) % 24
    else:
        # Winter (EST): NY Close is 22:00 UTC. To map 22 to 0, we add 2.
        return (utc_hour + 2) % 24

def get_session_name(broker_hour):
    if ASIAN_OPEN_HOUR <= broker_hour < 8: return "ASIAN"
    elif 8 <= broker_hour < 14: return "LONDON"
    elif 14 <= broker_hour < 17: return "NY_LONDON_OVERLAP"
    elif 17 <= broker_hour < TRADING_END_HOUR: return "NEW_YORK"
    else: return "ROLLOVER"
