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
            if cached.get("System Cache Date") == curr_day:
                daily_news_data = cached.get("Macroeconomic Calendar", [])
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
        
        raw_data = response.json()
        
        # Completely restructure json to layman terms
        color_map = {
            "High": "🔴 High Threat Level (Red Folder)",
            "Medium": "🟠 Medium Threat Level (Orange Folder)",
            "Low": "🟡 Low Activity (Yellow Folder)",
            "Non": "⚪ Non-Economic"
        }
        
        formatted_data = []
        for e in raw_data:
            orig_impact = e.get("impact", "")
            
            try:
                dt = datetime.fromisoformat(e.get('date', ''))
                human_time = dt.strftime("%A, %B %d, %Y at %I:%M %p")
            except:
                human_time = e.get('date', 'Unknown')
                
            readable_event = {
                "Event Name": e.get("title", "Unknown"),
                "Currency Affected": e.get("country", "Unknown"),
                "Date and Time": human_time,
                "Danger Level": color_map.get(orig_impact, orig_impact),
                "Expected Result (Forecast)": e.get("forecast", ""),
                "Previous Result": e.get("previous", ""),
                "_bot_raw_date": e.get("date", ""),
                "_bot_raw_impact": orig_impact
            }
            formatted_data.append(readable_event)
            
        daily_news_data = formatted_data
        last_news_fetch_day = curr_day
        api_failed_lockdown = False
        
        os.makedirs("data", exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({
                "System Cache Date": curr_day, 
                "Notice": "This file contains the macroeconomic outlook for the entire trading week.",
                "Macroeconomic Calendar": daily_news_data
            }, f, indent=4)
            
        print(f"[SUCCESS] News Database Fetched & Cached Successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to fetch news API: {e}")
        api_failed_lockdown = True

def is_news_blackout():
    if not NEWS_KILL_SWITCH: return False
    if api_failed_lockdown: return True
    
    now_utc = datetime.now(timezone.utc)
    # The JSON array is cached under "Macroeconomic Calendar" if we read it from disk, 
    # but 'daily_news_data' is just the array array natively in RAM.
    for event in daily_news_data:
        impact = event.get('_bot_raw_impact')
        currency = event.get('Currency Affected')
        
        # Gold Sensitivity Filter
        is_critical = (currency == 'USD' and impact in ['High', 'Medium']) or \
                      (currency == 'ALL' and impact == 'High')

        if is_critical:
            try:
                event_time_utc = datetime.fromisoformat(event.get('_bot_raw_date')).astimezone(timezone.utc)
                diff = abs((now_utc - event_time_utc).total_seconds())
                if diff <= 1800: # 30 mins
                    return True
            except:
                pass
    return False
