import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import os
import sys

# Must add root path so core and engine are reachable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import *

MODEL_FILE = "vwap_anomaly_model.joblib"
HISTORIC_BARS = 200000  # Approx 3 years of M5 data

def get_ml_data(symbol, timeframe, n):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['tick_volume'] = df['tick_volume'].astype(float)
    return df

def train_vwap_brain():
    print(f"🚀 [VWAP ARMOR ARCHITECTURE] Booting Machine Learning Protocol...")
    
    if not mt5.initialize():
        print("❌ MT5 Init failed.")
        return
        
    print(f"📥 Scraping ~3 Years of M5 Anomalies ({HISTORIC_BARS} bars)...")
    df = get_ml_data(SYMBOL, mt5.TIMEFRAME_M5, HISTORIC_BARS)
    if df is None:
        print("❌ Failed to pull data from broker. Ensure MT5 is running.")
        mt5.shutdown()
        return

    print("🧠 Engine computing 3-Year Baseline VWAP distributions...")
    
    # Needs to match vwap_reversion strategy calculation perfectly
    temp_df = df.copy()
    temp_df.set_index('time', inplace=True)
    vwap_line = ta.vwap(temp_df['high'], temp_df['low'], temp_df['close'], temp_df['tick_volume'])
    df['VWAP'] = vwap_line.values
    diff = df['close'] - df['VWAP']
    std_dev = diff.rolling(window=100).std()
    
    # 3.0 SD Bands
    df['VWAP_Upper'] = df['VWAP'] + (std_dev * 3.0)
    df['VWAP_Lower'] = df['VWAP'] - (std_dev * 3.0)
    
    # Calculate Auxiliary Features to judge the crash physics
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    adx_obj = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_obj['ADX_14'] if adx_obj is not None else 0

    df['Velocity_1'] = df['close'].diff(1)
    df['Velocity_3'] = df['close'].diff(3)
    df['Candle_Size_ATR'] = (df['high'] - df['low']) / df['ATR']

    # Filter purely mathematical anomalies (+/- 3.0 SD breaches)
    # 1 if Buy Stretch, -1 if Sell Stretch, 0 if normal noise
    conditions = [
        df['low'] <= df['VWAP_Lower'],
        df['high'] >= df['VWAP_Upper']
    ]
    df['Anomaly_Trigger'] = np.select(conditions, [1, -1], default=0)

    # Memory Isolation: We ONLY want to train the AI on the exact candles where an anomaly mathematically triggered
    anomaly_df = df[df['Anomaly_Trigger'] != 0].copy()
    print(f"🎯 Black Swan Anomalies Isolated: {len(anomaly_df)} pure instances found.")

    if len(anomaly_df) < 50:
        print("🚫 Not enough historical anomalies found in the timeframe to safely train deep neural clusters. Expanding lookback recommended.")
        mt5.shutdown()
        return

    print("⚖️ Structuring Deep Labeling Results (Success vs Cascade Death)...")
    
    labels = []
    # Loop over the actual indices from the original dataframe where anomaly occurred
    for idx in anomaly_df.index:
        trigger_type = anomaly_df.loc[idx, 'Anomaly_Trigger']
        vwap_baseline = anomaly_df.loc[idx, 'VWAP']
        trigger_price = anomaly_df.loc[idx, 'close']
        
        # Look ahead exactly 10 candles structurally to see if it safely reverted or crashed further
        max_idx = min(idx + 10, len(df))
        future_window = df.loc[idx+1 : max_idx]
        
        if trigger_type == 1:  # Buy Anomaly
            # Target 1 = Success (Price reaches VWAP baseline)
            # Target 0 = Failed Knife (Price drops another 1 ATR into a cascade)
            if future_window['high'].max() >= vwap_baseline:
                labels.append(1)
            else:
                labels.append(0)
        elif trigger_type == -1: # Sell Anomaly
            if future_window['low'].min() <= vwap_baseline:
                labels.append(1)
            else:
                labels.append(0)
                
    anomaly_df['Target'] = labels
    
    # Drop NAs
    anomaly_df = anomaly_df.replace([np.inf, -np.inf], 0).dropna()
    
    features = ['RSI', 'ATR', 'ADX', 'Velocity_1', 'Velocity_3', 'Candle_Size_ATR']
    X = anomaly_df[features]
    y = anomaly_df['Target']
    
    # 80/20 train/test
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"🤖 Training Knife-Catcher AI over {len(X_train)} structural anomalies...")
    model = RandomForestClassifier(
        n_estimators=500, 
        max_depth=10, 
        min_samples_leaf=2, 
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("🧪 Simulating unseen Black Swan Crashes...")
    test_probs = model.predict_proba(X_test)
    max_probs = np.max(test_probs, axis=1)
    
    # AI strict veto threshold (> 52% Confidence to authorize trade, else veto)
    confident_idx = max_probs > 0.52
    y_filtered = y_test[confident_idx]
    preds = model.predict(X_test)[confident_idx]
    
    if len(y_filtered) > 0:
        acc = accuracy_score(y_filtered, preds)
        print("="*50)
        print(f"✅ VETO AI Accuracy Rating: {acc:.2%}")
        print(f"🛡️ Trades Authorized: {len(y_filtered)} / {len(y_test)} (The rest were proactively blocked as Traps)")
        print("="*50)
        
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), MODEL_FILE)
        joblib.dump(model, model_path)
        print(f"💾 Armor Successfully Forged. Neural network saved to {MODEL_FILE}")
    else:
        print("❌ AI determined complete uncertainty across all tests. Model vetoed.")

    mt5.shutdown()

if __name__ == "__main__":
    train_vwap_brain()
