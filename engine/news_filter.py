import os
import json
import requests
import time
from datetime import datetime, timezone
from core.config import *

daily_news_data = []
last_news_fetch_day = -1
api_failed_lockdown = False

def fetch_economic_news():
    global daily_news_data, last_news_fetch_day, api_failed_lockdown
    curr_day = datetime.now().day
    if curr_day == last_news_fetch_day: return
    
    # Check local cache first
    cache_file = "data/news_cache.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            if cached.get("day") == curr_day:
                daily_news_data = cached.get("data", [])
                last_news_fetch_day = curr_day
                api_failed_lockdown = False
                print("\n[CACHE] Loaded ForexFactory Calendar from local cache.")
                return
        except: pass
    
    print(f"\n[NEWS] Fetching ForexFactory Macro Calendar from Database API...")
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        import random
        time.sleep(random.uniform(1.5, 3.0))
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() 
        
        daily_news_data = response.json()
        last_news_fetch_day = curr_day
        api_failed_lockdown = False
        
        os.makedirs("data", exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({"day": curr_day, "data": daily_news_data}, f)
            
        print(f"[SUCCESS] News Database Fetched & Cached Successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to fetch news API: {e}")
        api_failed_lockdown = True

def is_news_blackout():
    if not NEWS_KILL_SWITCH: return False
    if api_failed_lockdown: return True
    
    now_utc = datetime.now(timezone.utc)
    for event in daily_news_data:
        if event.get('country') == 'USD' and event.get('impact') == 'High':
            try:
                event_time_utc = datetime.fromisoformat(event.get('date')).astimezone(timezone.utc)
                diff = abs((now_utc - event_time_utc).total_seconds())
                if diff <= 1800: # 30 mins
                    return True
            except:
                pass
    return False
