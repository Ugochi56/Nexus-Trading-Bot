from strategies.base import BaseStrategy
from core.config import *
import numpy as np
import time

class SMCStrategy(BaseStrategy):
    def __init__(self, ai_model=None):
        super().__init__("SMC")
        self.trend_model = ai_model
        self.active_fvg = None
        self.last_traded_fvg_id = None
        self.last_denied_fvg_time = None
        self.ai_throttle_timer = 0
        self.dynamic_sl_padding = 0

    def get_trend_direction(self, df_trend):
        ema_50 = df_trend['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        close = df_trend.iloc[-1]['close']
        if close > ema_50: return 'UP'
        if close < ema_50: return 'DOWN'
        return 'FLAT'

    def find_fresh_fvg(self, df, direction, lookback=30):
        if len(df) < lookback + 2: return None
        fvgs = []
        for i in range(len(df) - lookback, len(df) - 2):
            c1, c3 = df.iloc[i], df.iloc[i+2]
            if direction == 'UP' and c3['low'] > c1['high'] and (c3['low'] - c1['high']) > MIN_GAP_SIZE:
                fvgs.append({'type': 'BUY', 'top': c3['low'], 'bottom': c1['high'], 'time': c3['time'], 'idx': i+2})
            elif direction == 'DOWN' and c1['low'] > c3['high'] and (c1['low'] - c3['high']) > MIN_GAP_SIZE:
                fvgs.append({'type': 'SELL', 'top': c1['low'], 'bottom': c3['high'], 'time': c3['time'], 'idx': i+2})
                
        for fvg in reversed(fvgs):
            is_broken = False
            for j in range(fvg['idx'] + 1, len(df)):
                fc = df.iloc[j]
                if fvg['type'] == 'BUY' and fc['close'] < fvg['bottom']:
                    is_broken = True; break
                if fvg['type'] == 'SELL' and fc['close'] > fvg['top']:
                    is_broken = True; break
            if not is_broken: 
                return fvg
        return None

    def check_macro_confluence(self, df_m15, df_h1, df_h4):
        if df_m15 is None or df_h1 is None or df_h4 is None: return "NEUTRAL"
        try:
            m15_ema = df_m15['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            h1_ema = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            h4_ema200 = df_h4['close'].ewm(span=200, adjust=False).mean().iloc[-1]
            
            m15_dir = 'UP' if df_m15.iloc[-1]['close'] > m15_ema else 'DOWN'
            h1_dir = 'UP' if df_h1.iloc[-1]['close'] > h1_ema else 'DOWN'
            h4_dir = 'UP' if df_h4.iloc[-1]['close'] > h4_ema200 else 'DOWN'
            
            if m15_dir == 'UP' and h1_dir == 'UP' and h4_dir == 'UP': return "UP"
            if m15_dir == 'DOWN' and h1_dir == 'DOWN' and h4_dir == 'DOWN': return "DOWN"
            return "NEUTRAL"
        except:
            return "NEUTRAL"

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

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        self.dynamic_sl_padding = atr * SL_ATR_MULTIPLIER
        trend = self.get_trend_direction(df_h1)
        new_fvg = self.find_fresh_fvg(df_m5, trend, lookback=30)
        
        signal_payload = None
        action_msg = f"[SCAN] {trend}"

        if new_fvg and new_fvg['time'] not in [self.last_traded_fvg_id, self.last_denied_fvg_time]:
            if self.active_fvg is None or new_fvg['time'] != self.active_fvg['time']:
                self.active_fvg = new_fvg
                print(f"\n[SMC] [NEW ZONE]: {self.active_fvg['type']} {self.active_fvg['bottom']:.2f}-{self.active_fvg['top']:.2f}")

        if self.active_fvg:
            action_msg = f"[WATCH] {self.active_fvg['type']}"
            if self.active_fvg['type'] == 'BUY':
                if current_price < self.active_fvg['bottom']: 
                    print(f"[SMC] [ZONE BROKEN]: BUY {self.active_fvg['bottom']:.2f}-{self.active_fvg['top']:.2f}")
                    self.active_fvg = None
                
                elif current_price <= self.active_fvg['top'] and (time.time() - self.ai_throttle_timer > 3.0):
                    self.ai_throttle_timer = time.time()
                    macro_dir = self.check_macro_confluence(df_adx, df_h1, df_h4)
                    
                    if macro_dir == 'DOWN':
                        sl = self.active_fvg['top'] + self.dynamic_sl_padding
                        print(f"\n[SMC] [MACRO TRAP DETECTED]: Inverting M5 BUY into Macro SELL")
                        signal_payload = {'signal': 'SELL', 'sl': sl, 'confidence': 0.99, 'comment': "TRAP SELL"}
                        self.last_traded_fvg_id = self.active_fvg['time']
                        self.active_fvg = None
                    else:
                        ai_verdict, ai_conf = self.get_trend_ai_permission(df_m5) 
                        if ai_verdict == 'BUY':
                            sl = self.active_fvg['bottom'] - self.dynamic_sl_padding
                            signal_payload = {'signal': 'BUY', 'sl': sl, 'confidence': ai_conf, 'comment': f"SMC AI:{ai_conf:.2f}"}
                            self.last_traded_fvg_id = self.active_fvg['time']
                            self.active_fvg = None
                        else:
                            if ai_verdict == 'SELL': reason = "Predicts trend reversal (DOWN)"
                            else: reason = f"Uncertain market (Score < {AI_CONFIDENCE_THRESHOLD})"
                            spam_key = f"BUY_{self.active_fvg['bottom']}_{reason}"
                            if getattr(self, 'last_spam', '') != spam_key:
                                print(f"[SMC] [AI DENIED BUY] ({ai_conf:.2f}) -> Reason: {reason}")
                                self.last_spam = spam_key
                            self.last_denied_fvg_time = self.active_fvg['time']
                            self.active_fvg = None 

            elif self.active_fvg['type'] == 'SELL':
                if current_price > self.active_fvg['top']: 
                    print(f"[SMC] [ZONE BROKEN]: SELL {self.active_fvg['bottom']:.2f}-{self.active_fvg['top']:.2f}")
                    self.active_fvg = None
                    
                elif current_price >= self.active_fvg['bottom'] and (time.time() - self.ai_throttle_timer > 3.0):
                    self.ai_throttle_timer = time.time()
                    macro_dir = self.check_macro_confluence(df_adx, df_h1, df_h4)
                    
                    if macro_dir == 'UP':
                        sl = self.active_fvg['bottom'] - self.dynamic_sl_padding
                        print(f"\n[SMC] [MACRO TRAP DETECTED]: Inverting M5 SELL into Macro BUY")
                        signal_payload = {'signal': 'BUY', 'sl': sl, 'confidence': 0.99, 'comment': "TRAP BUY"}
                        self.last_traded_fvg_id = self.active_fvg['time']
                        self.active_fvg = None
                    else:
                        ai_verdict, ai_conf = self.get_trend_ai_permission(df_m5) 
                        if ai_verdict == 'SELL':
                            sl = self.active_fvg['top'] + self.dynamic_sl_padding
                            signal_payload = {'signal': 'SELL', 'sl': sl, 'confidence': ai_conf, 'comment': f"SMC AI:{ai_conf:.2f}"}
                            self.last_traded_fvg_id = self.active_fvg['time']
                            self.active_fvg = None
                        else:
                            if ai_verdict == 'BUY': reason = "Predicts trend reversal (UP)"
                            else: reason = f"Uncertain market (Score < {AI_CONFIDENCE_THRESHOLD})"
                            spam_key = f"SELL_{self.active_fvg['top']}_{reason}"
                            if getattr(self, 'last_spam', '') != spam_key:
                                print(f"[SMC] [AI DENIED SELL] ({ai_conf:.2f}) -> Reason: {reason}")
                                self.last_spam = spam_key
                            self.last_denied_fvg_time = self.active_fvg['time']
                            self.active_fvg = None

        return {'payload': signal_payload, 'ui': action_msg}
