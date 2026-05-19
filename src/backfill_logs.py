import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
import os
from datetime import datetime, timedelta

mt5.initialize()

symbol = "XAUUSDm"

def get_historical_features(end_time):
    # Fetch M5 data up to the exact trade time
    rates_m5 = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, end_time, 250)
    rates_h1 = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, end_time, 100)
    rates_h4 = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H4, end_time, 100)
    
    if rates_m5 is None or len(rates_m5) == 0 or rates_h1 is None or len(rates_h1) == 0 or rates_h4 is None or len(rates_h4) == 0:
        return None
        
    df_m5 = pd.DataFrame(rates_m5)
    df_h1 = pd.DataFrame(rates_h1)
    df_h4 = pd.DataFrame(rates_h4)
    
    df_m5['close'] = df_m5['close'].astype(float)
    df_m5['high'] = df_m5['high'].astype(float)
    df_m5['low'] = df_m5['low'].astype(float)
    
    df_m5['EMA_50'] = df_m5.ta.ema(length=50)
    df_m5['EMA_200'] = df_m5.ta.ema(length=200)
    df_m5['RSI'] = df_m5.ta.rsi(length=14)
    df_m5['ATR'] = df_m5.ta.atr(length=14)
    adx = df_m5.ta.adx(length=14)
    df_m5['ADX'] = adx['ADX_14'] if adx is not None else 0
    
    latest = df_m5.iloc[-1:].copy()
    latest['Dist_EMA_50'] = (latest['close'] - latest['EMA_50']) / latest['close']
    latest['Dist_EMA_200'] = (latest['close'] - latest['EMA_200']) / latest['close']
    latest['Candle_Size'] = (latest['high'] - latest['low'])
    latest['Rel_Volatility'] = latest['Candle_Size'] / latest['ATR']
    latest['RSI_Zone'] = 1
    latest.loc[latest['RSI'] > 70, 'RSI_Zone'] = 2
    latest.loc[latest['RSI'] < 30, 'RSI_Zone'] = 0
    
    df_h1['RSI'] = df_h1.ta.rsi(length=14)
    df_h1['EMA_50'] = df_h1.ta.ema(length=50)
    df_h4_adx = df_h4.ta.adx(length=14)
    
    latest['H1_RSI'] = df_h1['RSI'].iloc[-1] if len(df_h1) > 0 else 50
    h1_ema_50 = df_h1['EMA_50'].iloc[-1] if len(df_h1) > 0 else latest['close'].values[0]
    latest['Dist_H1'] = (latest['close'] - h1_ema_50) / latest['close']
    latest['H4_ADX'] = df_h4_adx['ADX_14'].iloc[-1] if df_h4_adx is not None else 0
    
    latest.replace([np.inf, -np.inf], 0, inplace=True)
    latest.fillna(0, inplace=True)
    
    feature_cols = ['Dist_EMA_50', 'Dist_EMA_200', 'Dist_H1', 'RSI', 'RSI_Zone', 'Rel_Volatility', 'ADX', 'H1_RSI', 'H4_ADX']
    vec = latest[feature_cols].values[0]
    return vec

# Get deals from May 17 to today
start_date = datetime(2026, 5, 17)
end_date = datetime.now()
deals = mt5.history_deals_get(start_date, end_date)

if deals:
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    
    # We want entry deals (where trades were OPENED)
    entry_deals = df[df['entry'] == mt5.DEAL_ENTRY_IN]
    
    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'master_decision_log.csv')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    header_needed = not os.path.exists(log_file)
    
    recovered_count = 0
    with open(log_file, 'a') as f:
        if header_needed:
            f.write("Timestamp,Ticket,Signal,Comment,AI_Conf,Balance,Equity,Spread,Features\n")
            
        for _, row in entry_deals.iterrows():
            trade_time = datetime.fromtimestamp(row['time'])
            signal = "BUY" if row['type'] == mt5.DEAL_TYPE_BUY else "SELL"
            ticket = row['order'] # The order ticket that opened the position
            
            # Reconstruct the feature vector exactly as it was at that timestamp
            vec = get_historical_features(trade_time)
            if vec is not None:
                feat_str = "|".join([f"{v:.6f}" for v in vec])
                
                # We approximate AI_Conf, Balance, Equity, Spread since they weren't recorded
                # This is enough to let the Evolution Engine train on the PnL outcome
                fake_ai_conf = 0.70
                fake_balance = 100.00
                fake_equity = 100.00
                fake_spread = 15.0
                
                data_str = f"{trade_time},{ticket},{signal},RECOVERED_TRADE,{fake_ai_conf:.4f},{fake_balance:.2f},{fake_equity:.2f},{fake_spread:.1f},{feat_str}\n"
                f.write(data_str)
                recovered_count += 1
                
    print(f"Successfully recovered {recovered_count} historical trades and backfilled master_decision_log.csv!")
else:
    print("No deals found to recover.")

mt5.shutdown()
