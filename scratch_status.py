import sys
import os
import joblib
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import *
from engine.mt5_interface import connect_mt5, get_market_data
from core.indicators import calculate_rsi_simple, calculate_adx_simple
from core.utils import get_session_name

def main():
    connect_mt5()

    print(f"Fetching live market data for {SYMBOL}...")
    df_h1 = get_market_data(SYMBOL, TIMEFRAME_TREND, n=600)
    df_m5 = get_market_data(SYMBOL, TIMEFRAME_ENTRY, n=600)
    df_adx = get_market_data(SYMBOL, TIMEFRAME_ADX, n=100)

    if df_h1 is None or df_m5 is None:
        print("Market data unavailable. Are the markets closed for the weekend?")
        return

    current_price = df_m5['close'].iloc[-1]
    
    # H1 Trend Calculation
    import pandas_ta as ta
    df_h1['EMA_50'] = df_h1.ta.ema(length=50)
    h1_ema = df_h1['EMA_50'].iloc[-1]
    trend = "UP (Bullish)" if current_price > h1_ema else "DOWN (Bearish)"
    
    # Technical Indicators
    rsi = calculate_rsi_simple(df_m5['close'], RSI_PERIOD).iloc[-1]
    adx = calculate_adx_simple(df_adx, ADX_PERIOD).iloc[-1]
    
    # Time/Session
    last_tick_time = df_m5['time'].iloc[-1]
    session = get_session_name(last_tick_time.hour)
    
    # AI Prediction
    ai_up, ai_down = 0, 0
    try:
        trend_model = joblib.load(TREND_MODEL_FILE)
        df_m5['EMA_50'] = df_m5.ta.ema(length=50)
        df_m5['EMA_200'] = df_m5.ta.ema(length=200)
        df_m5['RSI'] = df_m5.ta.rsi(length=14)
        df_m5['ATR'] = df_m5.ta.atr(length=14)
        adx_series = df_m5.ta.adx(length=14)
        df_m5['ADX'] = adx_series['ADX_14'] if adx_series is not None else 0
        
        latest = df_m5.iloc[-2:].copy()
        latest['Dist_EMA_50'] = (latest['close'] - latest['EMA_50']) / latest['close']
        latest['Dist_EMA_200'] = (latest['close'] - latest['EMA_200']) / latest['close']
        
        df_m5['EMA_H1_Proxy'] = df_m5.ta.ema(length=600)
        latest['Dist_H1'] = (latest['close'] - df_m5['EMA_H1_Proxy'].iloc[-1]) / latest['close']
        latest['Candle_Size'] = (latest['high'] - latest['low'])
        latest['Rel_Volatility'] = latest['Candle_Size'] / latest['ATR']
        latest['RSI_Zone'] = 1
        latest.loc[latest['RSI'] > 70, 'RSI_Zone'] = 2
        latest.loc[latest['RSI'] < 30, 'RSI_Zone'] = 0
        latest.fillna(0, inplace=True)
        
        feature_cols = ['Dist_EMA_50', 'Dist_EMA_200', 'Dist_H1', 'RSI', 'RSI_Zone', 'Rel_Volatility', 'ADX']
        X_live = latest[feature_cols].iloc[[-1]]
        probs = trend_model.predict_proba(X_live)
        ai_down = probs[0][0] * 100
        ai_up = probs[0][1] * 100
    except Exception as e:
        print(f"Warning: AI Model not fully parsed for diagnostic script ({e})")

    print("\n=============================================")
    print("        NEXUS LIVE MARKET DIAGNOSTIC         ")
    print("=============================================")
    print(f"Time (Broker): {last_tick_time} | Session: {session}")
    print(f"Asset:         {SYMBOL}")
    print(f"Price:         ${current_price:.2f}")
    print(f"Macro Trend:   {trend} (H1 EMA: {h1_ema:.2f})")
    print("---------------------------------------------")
    if adx < 20:   adx_state = "Ranging/Chop"
    elif adx < 40: adx_state = "Trending"
    else:          adx_state = "Strong Trend"
    print(f"Volatility (ADX): {adx:.1f} ({adx_state})")
    
    if rsi > 70:   rsi_state = "Overbought"
    elif rsi < 30: rsi_state = "Oversold"
    else:          rsi_state = "Neutral"
    print(f"Momentum (RSI):   {rsi:.1f} ({rsi_state})")
    print("---------------------------------------------")
    print("AI Neural Net Current Prediction:")
    print(f"Probability of UP:   {ai_up:.1f}%")
    print(f"Probability of DOWN: {ai_down:.1f}%")
    
    print("=============================================\n")

if __name__ == "__main__":
    main()
