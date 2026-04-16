import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import time
import os
import sys

# Must add root path so core and engine are reachable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import *

MODEL_FILE = "gold_trend_model.joblib"
HISTORIC_BARS = 20000

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
    return df

def train_trend_brain(df_raw):
    print("\n🧠 1. Structuring Trend Features...")
    import pandas_ta as ta
    df = df_raw.copy()
    
    # Core Live Labeling (Look ahead 4 candles to see if trend went UP or DOWN)
    df['Future_Close'] = df['close'].shift(-4)
    df.dropna(inplace=True)
    
    # 1 if price went UP, 0 if price went DOWN
    df['Target'] = np.where(df['Future_Close'] > df['close'], 1, 0)
    
    features = ['Dist_EMA_50', 'Dist_EMA_200', 'Dist_H1', 'RSI', 'RSI_Zone', 'Rel_Volatility', 'ADX']
    
    X = df[features]
    y = df['Target']
    
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"🤖 Training Deep Forest Array on {len(X_train)} instances...")
    model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=12, 
        min_samples_split=10, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("🧪 Validating Accuracy against unseen blind data...")
    test_probs = model.predict_proba(X_test)
    max_probs = np.max(test_probs, axis=1)
    
    # Only test trades where the AI was > 55% confident
    confident_idx = max_probs > AI_CONFIDENCE_THRESHOLD
    y_filtered = y_test[confident_idx]
    preds = model.predict(X_test)[confident_idx]
    
    if len(y_filtered) > 0:
        acc = accuracy_score(y_filtered, preds)
        print("="*40)
        print(f"✅ Trend Model Accuracy: {acc:.2%}")
        print(f"📊 Trades Setup Passed: {len(y_filtered)} / {len(y_test)}")
        print("="*40)
        
        if acc >= 0.52:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), MODEL_FILE)
            joblib.dump(model, model_path)
            print(f"💾 Evolution Complete. {MODEL_FILE} permanently overwritten!")
        else:
            print("⚠️ Accuracy too low (Below 52% edge). Model Update REJECTED.")
    else:
        print("❌ No confidence setups identified. Skipping build.")


def train_reversal_brain(df_raw):
    print("\n🧠 2. Structuring RSI Reversal Matrix...")
    df = df_raw.copy()
    
    # 1. Filter anomalies
    conditions = [
        df['RSI'] <= 30,
        df['RSI'] >= 70
    ]
    df['Anomaly_Trigger'] = np.select(conditions, [1, -1], default=0)
    anomaly_df = df[df['Anomaly_Trigger'] != 0].copy()
    
    if len(anomaly_df) < 50:
        print("🚫 Not enough RSI extremities found recently. Skipping Reversal model.")
        return

    print("⚖️ Deep Labeling Results (1.0 ATR Snap)...")
    labels = []
    
    # Use reset index to avoid datetime indexing offset errors
    anomaly_df = anomaly_df.reset_index()
    
    for i in range(len(anomaly_df)):
        trigger_type = anomaly_df.loc[i, 'Anomaly_Trigger']
        trigger_price = anomaly_df.loc[i, 'close']
        trigger_atr = anomaly_df.loc[i, 'ATR']
        original_idx = anomaly_df.loc[i, 'index']
        
        # Look ahead 15 candles (1 hour 15 mins) safely using structural index
        max_idx = min(original_idx + 15, len(df))
        future_window = df.loc[original_idx+1 : max_idx]
        
        if trigger_type == 1:  # Oversold (BUY)
            if future_window['high'].max() >= trigger_price + (trigger_atr * 1.0):
                labels.append(1)
            else: labels.append(0)
        else: # Overbought (SELL)
            if future_window['low'].min() <= trigger_price - (trigger_atr * 1.0):
                labels.append(1)
            else: labels.append(0)
            
    anomaly_df['Target'] = labels
    features = ['Dist_EMA_50', 'Dist_EMA_200', 'Dist_H1', 'RSI', 'Rel_Volatility', 'ADX']
    
    X = anomaly_df[features]
    y = anomaly_df['Target']
    
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"🤖 Training Reversal Neural Net on {len(X_train)} squeezes...")
    model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train)
    
    test_probs = model.predict_proba(X_test)
    max_probs = np.max(test_probs, axis=1)
    confident_idx = max_probs > AI_CONFIDENCE_THRESHOLD
    y_filtered = y_test[confident_idx]
    preds = model.predict(X_test)[confident_idx]
    
    if len(y_filtered) > 0:
        acc = accuracy_score(y_filtered, preds)
        print("="*40)
        print(f"✅ Reversal Model Accuracy: {acc:.2%}")
        print(f"📊 Trades Setup Passed: {len(y_filtered)} / {len(y_test)}")
        print("="*40)
        
        if acc >= 0.52:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), REVERSAL_MODEL_FILE)
            joblib.dump(model, model_path)
            print(f"💾 Evolution Complete. {REVERSAL_MODEL_FILE} permanently overwritten!")
        else:
            print("⚠️ Accuracy too low (Below 52% edge). Reversal Model Update REJECTED.")
    else:
        print("❌ No confidence setups identified. Skipping build.")


def train_autonomous_brain():
    print("🚀 [NEXUS AUTO-RETRAINER] Booting Master Orchestrator...")
    
    if not mt5.initialize():
        print("❌ MT5 Init failed.")
        return
        
    print(f"📥 Downloading latest {HISTORIC_BARS} M5 candles from Broker...")
    df = get_ml_data(SYMBOL, mt5.TIMEFRAME_M5, HISTORIC_BARS)
    if df is None:
        print("❌ Failed to pull data.")
        mt5.shutdown()
        return

    import pandas_ta as ta
    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_200'] = df.ta.ema(length=200)
    df['EMA_H1_Proxy'] = df.ta.ema(length=600)
    df['RSI'] = df.ta.rsi(length=14)
    df['ATR'] = df.ta.atr(length=14)
    adx = df.ta.adx(length=14)
    df['ADX'] = adx['ADX_14']
    
    df['Dist_EMA_50'] = (df['close'] - df['EMA_50']) / df['close']
    df['Dist_EMA_200'] = (df['close'] - df['EMA_200']) / df['close']
    df['Dist_H1'] = (df['close'] - df['EMA_H1_Proxy']) / df['close']
    df['Candle_Size'] = (df['high'] - df['low'])
    df['Rel_Volatility'] = df['Candle_Size'] / df['ATR']
    df['RSI_Zone'] = 1
    df.loc[df['RSI'] > 70, 'RSI_Zone'] = 2
    df.loc[df['RSI'] < 30, 'RSI_Zone'] = 0

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    # 1. Train SMC Trend Filter
    train_trend_brain(df)

    # 2. Train RSI Mean-Reversion Model
    train_reversal_brain(df)

    mt5.shutdown()

if __name__ == "__main__":
    train_autonomous_brain()
