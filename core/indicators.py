import pandas as pd
import numpy as np
import pandas_ta as ta

def calculate_rsi_simple(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    if loss.iloc[-1] == 0: return 100
    return 100 - (100 / (1 + (gain / loss)))

def calculate_adx_simple(df, period=14):
    df = df.copy()
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['pdm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['ndm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['tr_s'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    df['pdm_s'] = pd.Series(df['pdm']).ewm(alpha=1/period, adjust=False).mean()
    df['ndm_s'] = pd.Series(df['ndm']).ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * (df['pdm_s'] / df['tr_s'])
    ndi = 100 * (df['ndm_s'] / df['tr_s'])
    dx = 100 * abs(pdi - ndi) / (pdi + ndi)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def calculate_atr_simple(df, period=14):
    high, low, close = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean().iloc[-1]

def calculate_poc(df, num_bins=50):
    """
    Calculates the Point of Control (POC) using Tick Volume.
    Divides the price range into `num_bins` and finds the bin with the highest volume.
    """
    if df is None or len(df) == 0 or 'tick_volume' not in df.columns:
        return 0.0
        
    min_price = df['low'].min()
    max_price = df['high'].max()
    
    if min_price == max_price:
        return min_price
        
    bins = np.linspace(min_price, max_price, num_bins + 1)
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    
    # Assign each typical price to a bin using cut
    try:
        binned = pd.cut(typical_price, bins=bins, include_lowest=True)
        # Sum tick volume by bin
        profile = df.groupby(binned, observed=False)['tick_volume'].sum()
        # Find the bin with max volume
        max_vol_bin = profile.idxmax()
        return max_vol_bin.mid
    except Exception as e:
        print(f"[ERROR] POC Calculation failed: {e}")
        return 0.0

def map_market_structure(df, lookback=2):
    """
    Calculates Pivot Highs and Lows (Fractals) to map market structure (HH, HL, LH, LL).
    lookback=2 means a pivot must be higher/lower than the 2 candles before and after it.
    Returns: A list of chronological structural pivot dictionaries.
    """
    pivots = []
    for i in range(lookback, len(df) - lookback):
        is_high = True
        is_low = True
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        
        for j in range(1, lookback + 1):
            if df.iloc[i-j]['high'] >= current_high or df.iloc[i+j]['high'] >= current_high:
                is_high = False
            if df.iloc[i-j]['low'] <= current_low or df.iloc[i+j]['low'] <= current_low:
                is_low = False
                
        if is_high:
            pivots.append({'type': 'HIGH', 'price': current_high, 'time': df.iloc[i]['time'], 'idx': i})
        if is_low:
            pivots.append({'type': 'LOW', 'price': current_low, 'time': df.iloc[i]['time'], 'idx': i})
            
    # Classify structural shifts
    structure = []
    last_high = None
    last_low = None
    
    for p in pivots:
        if p['type'] == 'HIGH':
            p['struct'] = 'HH' if (last_high is None or p['price'] > last_high['price']) else 'LH'
            last_high = p
            structure.append(p)
        elif p['type'] == 'LOW':
            p['struct'] = 'LL' if (last_low is None or p['price'] < last_low['price']) else 'HL'
            last_low = p
            structure.append(p)
            
    return structure
