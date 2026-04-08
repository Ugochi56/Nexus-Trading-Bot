from strategies.base import BaseStrategy
from core.config import *
import pandas_ta as ta
import numpy as np
import time

class VWAPReversionStrategy(BaseStrategy):
    def __init__(self, ai_model=None):
        super().__init__("VWAP_REVERSION")
        self.last_signal_time = None
        self.throttle_timer = 0
        self.sd_multiplier = 3.0  # 3 Standard Deviations for extreme anomaly stretches

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

        # Signal Logic
        if self.last_signal_time != last_time and (time.time() - self.throttle_timer > 3.0):
            # Scalp risk override: purely mathematical strat
            r_pct = min(current_risk, 0.5)
            sl_padding = atr * SL_ATR_MULTIPLIER

            if current_price >= upper_band:
                self.throttle_timer = time.time()
                self.last_signal_time = last_time
                signal_payload = {
                    'signal': 'SELL', 
                    'sl': current_price + sl_padding, 
                    'risk_override': r_pct, 
                    'confidence': 0.85, 
                    'comment': "VWAP Stretch SELL"
                }
                print(f"\n[VWAP] [EXTREME VOLUMETRIC STRETCH] (+{self.sd_multiplier} SD). Snapping Back.")
                
            elif current_price <= lower_band:
                self.throttle_timer = time.time()
                self.last_signal_time = last_time
                signal_payload = {
                    'signal': 'BUY', 
                    'sl': current_price - sl_padding, 
                    'risk_override': r_pct, 
                    'confidence': 0.85, 
                    'comment': "VWAP Stretch BUY"
                }
                print(f"\n[VWAP] [EXTREME VOLUMETRIC DEPRESSION] (-{self.sd_multiplier} SD). Snapping Back.")

        return {'payload': signal_payload, 'ui': action_msg}
