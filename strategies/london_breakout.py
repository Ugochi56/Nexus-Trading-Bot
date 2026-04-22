from strategies.base import BaseStrategy
from core.config import *
import pandas_ta as ta
import numpy as np
import time

class LondonBreakoutStrategy(BaseStrategy):
    def __init__(self, ai_model=None):
        super().__init__("LONDON_BREAKOUT")
        self.trend_model = ai_model
        self.asian_high = None
        self.asian_low = None
        self.last_signal_time = None
        self.throttle_timer = 0
        self.is_armed = False

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

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, server_hour=None, atr=None, **kwargs):
        if server_hour is None:
            return {'payload': None, 'ui': "[KILLZONE IDLE]"}

        action_msg = "[KILLZONE INACTIVE]"
        signal_payload = None

        # 1. Asian Accumulation Tracking
        if KILLZONE_TRACK_START <= server_hour < KILLZONE_TRACK_END:
            # We must map the highest/lowest inside the M5 dataframe that occurred within these hours
            # To be efficient, we track dynamically inside the evaluate loop
            self.is_armed = False
            last_candle = df_m5.iloc[-1]
            if self.asian_high is None or last_candle['high'] > self.asian_high:
                self.asian_high = last_candle['high']
            if self.asian_low is None or last_candle['low'] < self.asian_low:
                self.asian_low = last_candle['low']
                
            action_msg = f"[ASIAN MAPPING] H: {self.asian_high:.2f} | L: {self.asian_low:.2f}"
            return {'payload': None, 'ui': action_msg}

        # Reset geometry if we exceed the session bounds entirely
        if server_hour >= KILLZONE_EXEC_END or server_hour < KILLZONE_TRACK_START:
            self.asian_high = None
            self.asian_low = None
            self.is_armed = False
            return {'payload': None, 'ui': "[KILLZONE DEAD]"}

        # 2. Execution Window (London Killzone)
        if KILLZONE_EXEC_START <= server_hour < KILLZONE_EXEC_END:
            if self.asian_high is None or self.asian_low is None:
                return {'payload': None, 'ui': "[MISSING GEOMETRY]"}
                
            self.is_armed = True
            action_msg = f"[ARMED] Triggers H:{self.asian_high:.2f} L:{self.asian_low:.2f}"
            
            last_time = df_m5.iloc[-1]['time']
            if self.last_signal_time == last_time or (time.time() - self.throttle_timer < 3.0):
                return {'payload': None, 'ui': action_msg}

            # Macro Confluence Filtering (Institutional Fakeout Shield)
            macro_trend = "FLAT"
            if FILTER_LONDON_FAKEOUTS and df_h1 is not None and len(df_h1) >= 50:
                ema_50 = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                close_h1 = df_h1.iloc[-1]['close']
                macro_trend = "UP" if close_h1 > ema_50 else "DOWN"

            # Execute Breakouts
            if current_price > self.asian_high:
                if FILTER_LONDON_FAKEOUTS and macro_trend == "DOWN":
                    print(f"\\n[KILLZONE] [FAKEOUT SHIELD] Ignoring Upside break. Macro Trend favors DOWN. Waiting for sweep.")
                    self.throttle_timer = time.time()
                    return {'payload': None, 'ui': "[JUDAS SHIELD ACTIVE]"}
                
                ai_verdict, ai_conf = self.get_trend_ai_permission(df_m5)
                if ai_verdict == 'BUY':
                    print(f"\\n[KILLZONE] [ASIAN HIGH BROKEN] Institutional Volatility Surging (BUY)")
                    signal_payload = {'signal': 'BUY', 'sl': self.asian_low, 'confidence': ai_conf, 'comment': f"KZ BUY AI:{ai_conf:.2f}"}
                    self.last_signal_time = last_time
                    self.throttle_timer = time.time()
                else:
                    print(f"\\n[KILLZONE] [AI VETO] Breakout rejected by Neural Net ({ai_conf:.2f})")
                    self.throttle_timer = time.time()
                    
            elif current_price < self.asian_low:
                if FILTER_LONDON_FAKEOUTS and macro_trend == "UP":
                    print(f"\\n[KILLZONE] [FAKEOUT SHIELD] Ignoring Downside break. Macro Trend favors UP. Waiting for sweep.")
                    self.throttle_timer = time.time()
                    return {'payload': None, 'ui': "[JUDAS SHIELD ACTIVE]"}
                
                ai_verdict, ai_conf = self.get_trend_ai_permission(df_m5)
                if ai_verdict == 'SELL':
                    print(f"\\n[KILLZONE] [ASIAN LOW BROKEN] Institutional Volatility Surging (SELL)")
                    signal_payload = {'signal': 'SELL', 'sl': self.asian_high, 'confidence': ai_conf, 'comment': f"KZ SELL AI:{ai_conf:.2f}"}
                    self.last_signal_time = last_time
                    self.throttle_timer = time.time()
                else:
                    print(f"\\n[KILLZONE] [AI VETO] Breakout rejected by Neural Net ({ai_conf:.2f})")
                    self.throttle_timer = time.time()

        return {'payload': signal_payload, 'ui': action_msg}
