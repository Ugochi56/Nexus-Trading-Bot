from datetime import datetime
from core.config import *

def is_us_dst(date_obj):
    """Calculates if the given date is inside US Daylight Saving Time."""
    year = date_obj.year
    march_start = datetime(year, 3, 8)
    march_start = march_start.replace(day=14 - march_start.weekday()) 
    nov_end = datetime(year, 11, 1)
    nov_end = nov_end.replace(day=nov_end.day + (6 - nov_end.weekday()) % 7) 
    return march_start <= date_obj < nov_end

def get_session_name(hour):
    if ASIAN_OPEN_HOUR <= hour < 8: return "ASIAN"
    elif 8 <= hour < 14: return "LONDON"
    elif 14 <= hour < 17: return "NY_LONDON_OVERLAP"
    elif 17 <= hour < TRADING_END_HOUR: return "NEW_YORK"
    else: return "ROLLOVER"
