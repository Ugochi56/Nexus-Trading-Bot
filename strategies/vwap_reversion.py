from strategies.base import BaseStrategy
from core.config import *
import pandas_ta as ta
import numpy as np
import time
from core.state_manager import nexus_state

class VWAPReversionStrategy(BaseStrategy):
    def __init__(self, ai_model=None):
        super().__init__("VWAP_REVERSION")
        self.vwap_model = ai_model
        self.last_signal_time = None
        self.throttle_timer = 0
        self.sd_multiplier = 3.0  # 3 Standard Deviations for extreme anomaly stretches
        self.anomaly_buy_locked = nexus_state.get('vwap_buy_locked', False)
        self.anomaly_sell_locked = nexus_state.get('vwap_sell_locked', False)

    def set_buy_lock(self, status):
        self.anomaly_buy_locked = status
        nexus_state.set('vwap_buy_locked', status)

    def set_sell_lock(self, status):
        self.anomaly_sell_locked = status
        nexus_state.set('vwap_sell_locked', status)

    def calculate_vwap_bands(self, df):
        df = df.copy()
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
            
        try:
            # Anchored VWAP (Daily)
            vwap = ta.vwap(df['high'], df['low'], df['close'], df['tick_volume'])
            df['VWAP'] = vwap
            
            # Calculate standard deviation of the difference between close and VWAP
            diff = df['close'] - df['VWAP']
            std_dev = diff.rolling(window=100).std()
            
            df['VWAP_Upper'] = df['VWAP'] + (std_dev * self.sd_multiplier)
            df['VWAP_Lower'] = df['VWAP'] - (std_dev * self.sd_multiplier)
            return df
        except Exception as e:
            return None

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        signal_payload = None
        action_msg = "[VWAP_TRACK]"
        
        df_vwap = self.calculate_vwap_bands(df_m5)
        if df_vwap is None or len(df_vwap) < 100:
            return {'payload': None, 'ui': "[VWAP_INIT]"}
            
        latest = df_vwap.iloc[-1]
        vwap_val = latest.get('VWAP', 0)
        upper_band = latest.get('VWAP_Upper', 0)
        lower_band = latest.get('VWAP_Lower', 0)
        last_time = df_m5.iloc[-1]['time']
        
        if vwap_val == 0 or np.isnan(upper_band) or np.isnan(lower_band):
            return {'payload': None, 'ui': "[VWAP_WAIT]"}
            
        # UI Action Feedback string
        dist_pct = abs((current_price - vwap_val) / vwap_val) * 100
        action_msg = f"[vDiv]: {dist_pct:.2f}%"

        # Dynamic Normalization Reset
        if current_price >= vwap_val and self.anomaly_buy_locked:
            self.set_buy_lock(False)
            print("\n[VWAP] [ANOMALY RESOLVED] Asset normalized above VWAP. Buy triggers unlocked.")
        if current_price <= vwap_val and self.anomaly_sell_locked:
            self.set_sell_lock(False)
            print("\n[VWAP] [ANOMALY RESOLVED] Asset normalized below VWAP. Sell triggers unlocked.")

        # Signal Logic
        if self.last_signal_time != last_time and (time.time() - self.throttle_timer > 3.0):
            # Scalp risk override: purely mathematical strat
            r_pct = min(current_risk, 0.5)
            sl_padding = atr * SL_ATR_MULTIPLIER

            if current_price >= upper_band and not self.anomaly_sell_locked:
                if self.vwap_model and USE_AI_FILTER:
                    try:
                        rsi = ta.rsi(df_m5['close'], length=14).iloc[-1]
                        atr = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=14).iloc[-1]
                        adx_obj = ta.adx(df_m5['high'], df_m5['low'], df_m5['close'], length=14)
                        adx = adx_obj['ADX_14'].iloc[-1] if adx_obj is not None else 0
                        v1 = df_m5['close'].diff(1).iloc[-1]
                        v3 = df_m5['close'].diff(3).iloc[-1]
                        c_atr = (df_m5['high'].iloc[-1] - df_m5['low'].iloc[-1]) / atr
                        features = [[rsi, atr, adx, v1, v3, c_atr]]
                        
                        probs = self.vwap_model.predict_proba(features)[0]
                        if probs[1] < AI_CONFIDENCE_THRESHOLD:
                            print("\r[VWAP] [AI VETO] Machine Learning predicted cascading continuation past +3.0 SD. Sell Trade blocked.".ljust(90), end='')
                            return {'payload': None, 'ui': "[VWAP_VETO] CASC"}
                    except Exception as e:
                        pass
                        
                self.throttle_timer = time.time()
                self.last_signal_time = last_time
                self.set_sell_lock(True)
                signal_payload = {
                    'signal': 'SELL', 
                    'sl': current_price + sl_padding, 
                    'risk_override': r_pct, 
                    'confidence': 0.85, 
                    'comment': "VWAP Stretch SELL"
                }
                print(f"\n[VWAP] [EXTREME VOLUMETRIC STRETCH] (+{self.sd_multiplier} SD). Snapping Back.")
                
            elif current_price <= lower_band and not self.anomaly_buy_locked:
                if self.vwap_model and USE_AI_FILTER:
                    try:
                        rsi = ta.rsi(df_m5['close'], length=14).iloc[-1]
                        atr = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=14).iloc[-1]
                        adx_obj = ta.adx(df_m5['high'], df_m5['low'], df_m5['close'], length=14)
                        adx = adx_obj['ADX_14'].iloc[-1] if adx_obj is not None else 0
                        v1 = df_m5['close'].diff(1).iloc[-1]
                        v3 = df_m5['close'].diff(3).iloc[-1]
                        c_atr = (df_m5['high'].iloc[-1] - df_m5['low'].iloc[-1]) / atr
                        features = [[rsi, atr, adx, v1, v3, c_atr]]
                        
                        probs = self.vwap_model.predict_proba(features)[0]
                        if probs[1] < AI_CONFIDENCE_THRESHOLD:
                            print("\r[VWAP] [AI VETO] Machine Learning predicted cascading crash past -3.0 SD. Knife-Catch blocked.".ljust(90), end='')
                            return {'payload': None, 'ui': "[VWAP_VETO] CASC"}
                    except Exception as e:
                        pass
                        
                self.throttle_timer = time.time()
                self.last_signal_time = last_time
                self.set_buy_lock(True)
                signal_payload = {
                    'signal': 'BUY', 
                    'sl': current_price - sl_padding, 
                    'risk_override': r_pct, 
                    'confidence': 0.85, 
                    'comment': "VWAP Stretch BUY"
                }
                print(f"\n[VWAP] [EXTREME VOLUMETRIC DEPRESSION] (-{self.sd_multiplier} SD). Snapping Back.")

        return {'payload': signal_payload, 'ui': action_msg}
