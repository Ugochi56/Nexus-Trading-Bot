from strategies.base import BaseStrategy
from core.config import *
import pandas_ta as ta
import numpy as np
import time

class RSIReversionStrategy(BaseStrategy):
    def __init__(self, ai_model=None):
        super().__init__("RSI_REVERSION")
        self.reversal_model = ai_model
        self.last_rsi_signal_time = None
        self.ai_throttle_timer = 0
        self.dynamic_sl_padding = 0

    def get_reversal_ai_permission(self, df_m5):
        if not USE_AI_FILTER or self.reversal_model is None:
            return 'SKIP_CHECK', 0.0
        try:
            df = df_m5.copy()
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['RSI'] = ta.rsi(df['close'], length=14)
            df['EMA_50'] = ta.ema(df['close'], length=50)
            
            bbands = ta.bbands(df['close'], length=20, std=2.0)
            bbu_col = [c for c in bbands.columns if c.startswith('BBU')][0]
            bbl_col = [c for c in bbands.columns if c.startswith('BBL')][0]
            df['BB_Upper'] = bbands[bbu_col]
            df['BB_Lower'] = bbands[bbl_col]
            
            df['Dist_BB_Upper'] = (df['high'] - df['BB_Upper']) / (df['ATR'] + 0.0001)
            df['Dist_BB_Lower'] = (df['low'] - df['BB_Lower']) / (df['ATR'] + 0.0001)
            
            df['Vol_MA'] = df['tick_volume'].rolling(20).mean()
            df['Rel_Volume'] = df['tick_volume'] / (df['Vol_MA'] + 0.0001) 
            
            df['Upper_Wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['Lower_Wick'] = df[['open', 'close']].min(axis=1) - df['low']
            df['Upper_Wick_Ratio'] = df['Upper_Wick'] / (df['ATR'] + 0.0001)
            df['Lower_Wick_Ratio'] = df['Lower_Wick'] / (df['ATR'] + 0.0001)
            df['Stretch_50'] = (df['close'] - df['EMA_50']) / (df['ATR'] + 0.0001)
            
            df.dropna(inplace=True)
            if len(df) == 0: return 'SKIP_CHECK', 0.0
            
            latest = df.iloc[[-1]]
            features = ['RSI', 'Dist_BB_Upper', 'Dist_BB_Lower', 'Rel_Volume', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio', 'Stretch_50']
            X_live = latest[features]
            
            probs = self.reversal_model.predict_proba(X_live)
            prob_buy = probs[0][1]
            prob_sell = probs[0][2]
            
            if prob_buy >= AI_CONFIDENCE_THRESHOLD: return 'BUY', prob_buy
            elif prob_sell >= AI_CONFIDENCE_THRESHOLD: return 'SELL', prob_sell
            else: return 'UNCERTAIN', max(prob_buy, prob_sell)
        except Exception as e:
            return 'SKIP_CHECK', 0.0

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        self.dynamic_sl_padding = atr * SL_ATR_MULTIPLIER
        signal_payload = None
        action_msg = "[SCALP] Dual AI"

        last_candle_time = df_m5.iloc[-1]['time']
        curr_rsi = df_m5.iloc[-1].get('rsi', 50) 
        # Note: RSI is calculated dynamically externally or here? It's passed via df_m5 if calculated by core
        if 'rsi' not in df_m5.columns:
            df_m5['rsi'] = ta.rsi(df_m5['close'], length=14)
            curr_rsi = df_m5.iloc[-1]['rsi']

        scalp_risk = min(current_risk, 0.5) 

        if self.last_rsi_signal_time != last_candle_time and (time.time() - self.ai_throttle_timer > 3.0):
            if curr_rsi <= RSI_OVERSOLD or curr_rsi >= RSI_OVERBOUGHT:
                self.ai_throttle_timer = time.time()
                ai_verdict, ai_conf = self.get_reversal_ai_permission(df_m5)
                
                if curr_rsi <= RSI_OVERSOLD and ai_verdict == 'BUY':
                    sl = current_price - self.dynamic_sl_padding
                    signal_payload = {'signal': 'BUY', 'sl': sl, 'risk_override': scalp_risk, 'confidence': ai_conf, 'comment': f"RSI-AI:{ai_conf:.2f}"}
                    self.last_rsi_signal_time = last_candle_time
                    
                elif curr_rsi >= RSI_OVERBOUGHT and ai_verdict == 'SELL':
                    sl = current_price + self.dynamic_sl_padding
                    signal_payload = {'signal': 'SELL', 'sl': sl, 'risk_override': scalp_risk, 'confidence': ai_conf, 'comment': f"RSI-AI:{ai_conf:.2f}"}
                    self.last_rsi_signal_time = last_candle_time

        return {'payload': signal_payload, 'ui': action_msg}
