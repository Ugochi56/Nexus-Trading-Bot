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
