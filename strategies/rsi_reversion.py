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

    def get_reversal_ai_permission(self, df_m5, df_h1, df_h4):
        if not USE_AI_FILTER or self.reversal_model is None:
            return 'SKIP_CHECK', 0.0
        try:
            df = df_m5.copy()
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['RSI'] = ta.rsi(df['close'], length=14)
            df['EMA_50'] = ta.ema(df['close'], length=50)
            df['EMA_200'] = ta.ema(df['close'], length=200)
            
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx is not None: df['ADX'] = adx['ADX_14']
            else: df['ADX'] = 0
            
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
            
            latest = df.iloc[[-1]].copy()
            
            # Legacy features matching data collector
            latest['Dist_EMA_50'] = (latest['close'] - latest['EMA_50']) / latest['close']
            latest['Dist_EMA_200'] = (latest['close'] - latest['EMA_200']) / latest['close']
            latest['Candle_Size'] = (latest['high'] - latest['low'])
            latest['Rel_Volatility'] = latest['Candle_Size'] / latest['ATR']
            
            if df_h1 is not None and df_h4 is not None:
                latest['H1_RSI'] = df_h1.ta.rsi(length=14).iloc[-1]
                h1_ema_50 = df_h1.ta.ema(length=50).iloc[-1]
                latest['Dist_H1'] = (latest['close'] - h1_ema_50) / latest['close']
                adx_h4 = df_h4.ta.adx(length=14)
                latest['H4_ADX'] = adx_h4['ADX_14'].iloc[-1] if adx_h4 is not None else 0
            else:
                latest['H1_RSI'] = 50
                latest['Dist_H1'] = 0
                latest['H4_ADX'] = 0

            latest.replace([np.inf, -np.inf], 0, inplace=True)
            latest.fillna(0, inplace=True)
            
            features = ['Dist_EMA_50', 'Dist_EMA_200', 'Dist_H1', 'RSI', 'Rel_Volatility', 'ADX', 'H1_RSI', 'H4_ADX']
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

        last_candle_time = df_m5.iloc[-1]['time']
        curr_rsi = df_m5.iloc[-1].get('rsi', 50) 
        # Note: RSI is calculated dynamically externally or here? It's passed via df_m5 if calculated by core
        if 'rsi' not in df_m5.columns:
            df_m5['rsi'] = ta.rsi(df_m5['close'], length=14)
            curr_rsi = df_m5.iloc[-1]['rsi']
            
        if 'sma_14' not in df_m5.columns:
            df_m5['sma_14'] = ta.sma(df_m5['close'], length=14)
            
        curr_sma = df_m5.iloc[-1].get('sma_14', current_price)

        action_msg = f"[RSI: {curr_rsi:.0f}]"

        scalp_risk = min(current_risk, 0.5) 

        if self.last_rsi_signal_time != last_candle_time and (time.time() - self.ai_throttle_timer > 3.0):
            if curr_rsi <= RSI_OVERSOLD or curr_rsi >= RSI_OVERBOUGHT:
                self.ai_throttle_timer = time.time()
                ai_verdict, ai_conf = self.get_reversal_ai_permission(df_m5, df_h1, df_h4)
                
                if curr_rsi <= RSI_OVERSOLD and ai_verdict == 'BUY':
                    sl = current_price - self.dynamic_sl_padding
                    signal_payload = {'signal': 'BUY', 'sl': sl, 'tp_price': curr_sma, 'risk_override': scalp_risk, 'confidence': ai_conf, 'comment': f"RSI-AI:{ai_conf:.2f}"}
                    self.last_rsi_signal_time = last_candle_time
                    
                elif curr_rsi >= RSI_OVERBOUGHT and ai_verdict == 'SELL':
                    sl = current_price + self.dynamic_sl_padding
                    signal_payload = {'signal': 'SELL', 'sl': sl, 'tp_price': curr_sma, 'risk_override': scalp_risk, 'confidence': ai_conf, 'comment': f"RSI-AI:{ai_conf:.2f}"}
                    self.last_rsi_signal_time = last_candle_time

        return {'payload': signal_payload, 'ui': action_msg}
