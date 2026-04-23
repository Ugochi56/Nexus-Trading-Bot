from strategies.base import BaseStrategy
from core.config import *
import numpy as np
import time

class SMCOrderBlockStrategy(BaseStrategy):
    def __init__(self, ai_model=None):
        super().__init__("SMC_ORDERBLOCK")
        self.trend_model = ai_model
        self.active_break_dir = None
        self.active_obs = []
        self.max_obs = 3
        self.last_traded_ob_time = None
        self.ai_throttle_timer = 0
        self.structural_lookback = 40  # Lookback for Swing High/Low

    def reset(self):
        self.active_obs = []
        self.active_break_dir = None
        self.last_traded_ob_time = None

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

    def get_trend_direction(self, df_h1, df_h4):
        if df_h4 is None or df_h1 is None: return 'FLAT'
        h1_ema_50 = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        h4_ema_200 = df_h4['close'].ewm(span=200, adjust=False).mean().iloc[-1]
        
        h1_close = df_h1.iloc[-1]['close']
        h4_close = df_h4.iloc[-1]['close']
        
        if h1_close > h1_ema_50 and h4_close > h4_ema_200: return 'UP'
        if h1_close < h1_ema_50 and h4_close < h4_ema_200: return 'DOWN'
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
        
        # Momentum Metric calculation
        atr = df['high'].iloc[-14:] - df['low'].iloc[-14:]
        avg_candle_size = atr.mean()
        
        ob = None

        # BULLISH BOS (Price breaks recent High)
        if current_candle['close'] > highest_high * 0.9995: 
            # 1. Momentum Validation: Impulse leg must be explosive (1.5x average size)
            if (current_candle['high'] - current_candle['low']) < (avg_candle_size * 1.5):
                return None 
                
            # Find the last Bearish (Red) candle before this impulse
            for i in range(len(df)-2, len(df) - 15, -1):
                if df.iloc[i]['close'] < df.iloc[i]['open']:
                    # 2. Zone Expansion: Pad block structurally encompassing previous tick
                    zone_high = max(df.iloc[i]['high'], df.iloc[i-1]['high'])
                    zone_low = min(df.iloc[i]['low'], df.iloc[i-1]['low'])
                    
                    ob = {
                        'type': 'BUY',
                        'top': zone_high,
                        'bottom': zone_low,
                        'time': df.iloc[i]['time'],
                        'bos_time': current_candle['time']
                    }
                    return ob

        # BEARISH BOS (Price breaks recent Low)
        elif current_candle['close'] < lowest_low * 1.0005:
            # Momentum Validation
            if (current_candle['high'] - current_candle['low']) < (avg_candle_size * 1.5):
                return None
                
            # Find the last Bullish (Green) candle before this impulse
            for i in range(len(df)-2, len(df) - 15, -1):
                if df.iloc[i]['close'] > df.iloc[i]['open']:
                    # Zone Expansion
                    zone_high = max(df.iloc[i]['high'], df.iloc[i-1]['high'])
                    zone_low = min(df.iloc[i]['low'], df.iloc[i-1]['low'])
                    
                    ob = {
                        'type': 'SELL',
                        'top': zone_high,
                        'bottom': zone_low,
                        'time': df.iloc[i]['time'],
                        'bos_time': current_candle['time']
                    }
                    return ob
                    
        return None

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        trend = self.get_trend_direction(df_h1, df_h4)
        signal_payload = None
        action_msg = f"[OB_SCAN] {trend}"
        
        # Structural SL padding (Fractional offset to avoid broker spread-hunting)
        # Using Structural Wick Low/High for primary SL, slightly padded by 0.2 ATR
        spread_padding = atr * 0.2

        new_ob = self.find_order_block(df_m5)
        
        if new_ob and new_ob['time'] != self.last_traded_ob_time:
            if not any(ob['time'] == new_ob['time'] for ob in self.active_obs):
                self.active_obs.append(new_ob)
                if len(self.active_obs) > self.max_obs:
                    self.active_obs.pop(0)
                print(f"[OB] [FRESH ORDER BLOCK]: {new_ob['type']} {new_ob['bottom']:.2f}-{new_ob['top']:.2f}")

        if self.active_obs:
            action_msg = f"[OB_WATCH] {len(self.active_obs)} Active Block(s)"
            
            valid_obs = []
            for ob in self.active_obs:
                is_valid = True
                
                if ob['type'] == 'BUY':
                    if current_price < ob['bottom']: 
                        print(f"[OB] [BLOCK INVALIDATED]: BUY {ob['bottom']:.2f} Penetrated/Mitigated.")
                        is_valid = False
                    
                    elif current_price <= ob['top'] and (time.time() - self.ai_throttle_timer > 3.0):
                        self.ai_throttle_timer = time.time()
                        ai_verdict, ai_conf = self.get_trend_ai_permission(df_m5) 
                        
                        if ai_verdict == 'BUY':
                            structural_sl = ob['bottom'] - spread_padding
                            signal_payload = {'signal': 'BUY', 'sl': structural_sl, 'confidence': ai_conf, 'comment': f"OB AI:{ai_conf:.2f}"}
                            self.last_traded_ob_time = ob['time']
                            is_valid = False 
                        else:
                            if ai_verdict == 'SELL': reason = "Predicts trend reversal (DOWN)"
                            else: reason = f"Uncertain market (Score < {AI_CONFIDENCE_THRESHOLD})"
                            spam_key = f"BUY_{ob['bottom']}_{reason}"
                            if getattr(self, 'last_spam', '') != spam_key:
                                print(f"[OB] [AI DENIED OB BUY] ({ai_conf:.2f}) -> {reason}")
                                self.last_spam = spam_key

                elif ob['type'] == 'SELL':
                    if current_price > ob['top']: 
                        print(f"[OB] [BLOCK INVALIDATED]: SELL {ob['top']:.2f} Penetrated/Mitigated.")
                        is_valid = False
                        
                    elif current_price >= ob['bottom'] and (time.time() - self.ai_throttle_timer > 3.0):
                        self.ai_throttle_timer = time.time()
                        ai_verdict, ai_conf = self.get_trend_ai_permission(df_m5) 
                        
                        if ai_verdict == 'SELL':
                            structural_sl = ob['top'] + spread_padding
                            signal_payload = {'signal': 'SELL', 'sl': structural_sl, 'confidence': ai_conf, 'comment': f"OB AI:{ai_conf:.2f}"}
                            self.last_traded_ob_time = ob['time']
                            is_valid = False
                        else:
                            if ai_verdict == 'BUY': reason = "Predicts trend reversal (UP)"
                            else: reason = f"Uncertain market (Score < {AI_CONFIDENCE_THRESHOLD})"
                            spam_key = f"SELL_{ob['top']}_{reason}"
                            if getattr(self, 'last_spam', '') != spam_key:
                                print(f"[OB] [AI DENIED OB SELL] ({ai_conf:.2f}) -> {reason}")
                                self.last_spam = spam_key
                
                if is_valid:
                    valid_obs.append(ob)
                    
            self.active_obs = valid_obs

        return {'payload': signal_payload, 'ui': action_msg}
