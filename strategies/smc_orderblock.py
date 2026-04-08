from strategies.base import BaseStrategy
from core.config import *
import numpy as np
import time

class SMCOrderBlockStrategy(BaseStrategy):
    def __init__(self, ai_model=None):
        super().__init__("SMC_ORDERBLOCK")
        self.trend_model = ai_model
        self.active_break_dir = None
        self.active_ob = None
        self.last_traded_ob_time = None
        self.ai_throttle_timer = 0
        self.structural_lookback = 40  # Lookback for Swing High/Low

    def get_trend_ai_permission(self, df_m5):
        if not USE_AI_FILTER or self.trend_model is None:
            return 'SKIP_CHECK', 0.0
        try:
            df = df_m5.copy()
            if 'EMA_50' not in df.columns:
                df['EMA_50'] = df.ta.ema(length=50)
                df['EMA_200'] = df.ta.ema(length=200)
                df['EMA_H1_Proxy'] = df.ta.ema(length=600)
                df['RSI'] = df.ta.rsi(length=14)
                df['ATR'] = df.ta.atr(length=14)
                adx = df.ta.adx(length=14)
                df['ADX'] = adx['ADX_14']

            latest = df.iloc[-2:].copy()
            latest['Dist_EMA_50'] = (latest['close'] - latest['EMA_50']) / latest['close']
            latest['Dist_EMA_200'] = (latest['close'] - latest['EMA_200']) / latest['close']
            latest['Dist_H1'] = (latest['close'] - latest['EMA_H1_Proxy']) / latest['close']
            latest['Candle_Size'] = (latest['high'] - latest['low'])
            latest['Rel_Volatility'] = latest['Candle_Size'] / latest['ATR']
            latest['RSI_Zone'] = 1
            latest.loc[latest['RSI'] > 70, 'RSI_Zone'] = 2
            latest.loc[latest['RSI'] < 30, 'RSI_Zone'] = 0

            latest.replace([np.inf, -np.inf], 0, inplace=True)
            latest.fillna(0, inplace=True)

            feature_cols = ['Dist_EMA_50', 'Dist_EMA_200', 'Dist_H1', 'RSI', 'RSI_Zone', 'Rel_Volatility', 'ADX']
            X_live = latest[feature_cols].iloc[[-1]]

            probs = self.trend_model.predict_proba(X_live)
            prob_down = probs[0][0]
            prob_up = probs[0][1]

            if prob_up >= AI_CONFIDENCE_THRESHOLD: return 'BUY', prob_up
            elif prob_down >= AI_CONFIDENCE_THRESHOLD: return 'SELL', prob_down
            else: return 'UNCERTAIN', max(prob_up, prob_down)
        except Exception as e:
            return 'SKIP_CHECK', 0.0

    def get_trend_direction(self, df_trend):
        ema_50 = df_trend['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        close = df_trend.iloc[-1]['close']
        if close > ema_50: return 'UP'
        if close < ema_50: return 'DOWN'
        return 'FLAT'

    def find_order_block(self, df):
        """
        Calculates recent Break of Structure (BOS) and identifies the resulting Order Block.
        """
        if len(df) < self.structural_lookback + 5:
            return None
            
        recent_chunk = df.iloc[-self.structural_lookback:]
        current_candle = df.iloc[-1]
        
        highest_high = recent_chunk['high'].max()
        lowest_low = recent_chunk['low'].min()
        
        ob = None

        # BULLISH BOS (Price breaks recent High)
        if current_candle['close'] > highest_high * 0.9995: 
            # Find the last Bearish (Red) candle before this impulse
            for i in range(len(df)-2, len(df) - 15, -1):
                if df.iloc[i]['close'] < df.iloc[i]['open']: # It's a Red Candle
                    ob = {
                        'type': 'BUY',
                        'top': df.iloc[i]['high'],
                        'bottom': df.iloc[i]['low'],
                        'time': df.iloc[i]['time'],
                        'bos_time': current_candle['time']
                    }
                    return ob

        # BEARISH BOS (Price breaks recent Low)
        elif current_candle['close'] < lowest_low * 1.0005:
            # Find the last Bullish (Green) candle before this impulse
            for i in range(len(df)-2, len(df) - 15, -1):
                if df.iloc[i]['close'] > df.iloc[i]['open']: # It's a Green Candle
                    ob = {
                        'type': 'SELL',
                        'top': df.iloc[i]['high'],
                        'bottom': df.iloc[i]['low'],
                        'time': df.iloc[i]['time'],
                        'bos_time': current_candle['time']
                    }
                    return ob
                    
        return None

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        trend = self.get_trend_direction(df_h1)
        signal_payload = None
        action_msg = f"[OB_SCAN] {trend}"
        
        # Structural SL padding (Fractional offset to avoid broker spread-hunting)
        # Using Structural Wick Low/High for primary SL, slightly padded by 0.2 ATR
        spread_padding = atr * 0.2

        new_ob = self.find_order_block(df_m5)
        
        if new_ob and new_ob['time'] != self.last_traded_ob_time:
            if self.active_ob is None or new_ob['time'] != self.active_ob['time']:
                self.active_ob = new_ob
                print(f"\n[OB] [FRESH ORDER BLOCK]: {self.active_ob['type']} {self.active_ob['bottom']:.2f}-{self.active_ob['top']:.2f}")

        if self.active_ob:
            action_msg = f"[OB_WATCH] {self.active_ob['type']}"
            
            if self.active_ob['type'] == 'BUY':
                if current_price < self.active_ob['bottom']: 
                    print(f"\n[OB] [BLOCK INVALIDATED]: BUY {self.active_ob['bottom']:.2f} Penetrated/Mitigated.")
                    self.active_ob = None
                
                # Execution Trigger: Price falls into the top of the Block
                elif current_price <= self.active_ob['top'] and (time.time() - self.ai_throttle_timer > 3.0):
                    self.ai_throttle_timer = time.time()
                    ai_verdict, ai_conf = self.get_trend_ai_permission(df_m5) 
                    
                    if ai_verdict == 'BUY':
                        # The SL is exactly 1 pip/fraction below the absolute bottom wick. Pure structural risk.
                        structural_sl = self.active_ob['bottom'] - spread_padding
                        signal_payload = {'signal': 'BUY', 'sl': structural_sl, 'confidence': ai_conf, 'comment': f"OB AI:{ai_conf:.2f}"}
                        self.last_traded_ob_time = self.active_ob['time']
                        self.active_ob = None
                    else:
                        if ai_verdict == 'SELL': reason = "Predicts trend reversal (DOWN)"
                        else: reason = f"Uncertain market (Score < {AI_CONFIDENCE_THRESHOLD})"
                        print(f"\n[OB] [AI DENIED OB BUY] ({ai_conf:.2f}) -> {reason}")

            elif self.active_ob['type'] == 'SELL':
                if current_price > self.active_ob['top']: 
                    print(f"\n[OB] [BLOCK INVALIDATED]: SELL {self.active_ob['top']:.2f} Penetrated/Mitigated.")
                    self.active_ob = None
                    
                # Execution Trigger: Price rises into the bottom of the Block
                elif current_price >= self.active_ob['bottom'] and (time.time() - self.ai_throttle_timer > 3.0):
                    self.ai_throttle_timer = time.time()
                    ai_verdict, ai_conf = self.get_trend_ai_permission(df_m5) 
                    
                    if ai_verdict == 'SELL':
                        # The SL is physically above the absolute top wick. Pure structural risk.
                        structural_sl = self.active_ob['top'] + spread_padding
                        signal_payload = {'signal': 'SELL', 'sl': structural_sl, 'confidence': ai_conf, 'comment': f"OB AI:{ai_conf:.2f}"}
                        self.last_traded_ob_time = self.active_ob['time']
                        self.active_ob = None
                    else:
                        if ai_verdict == 'BUY': reason = "Predicts trend reversal (UP)"
                        else: reason = f"Uncertain market (Score < {AI_CONFIDENCE_THRESHOLD})"
                        print(f"\n[OB] [AI DENIED OB SELL] ({ai_conf:.2f}) -> {reason}")

        return {'payload': signal_payload, 'ui': action_msg}
