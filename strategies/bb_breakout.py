from strategies.base import BaseStrategy
from core.config import *
import pandas_ta as ta
import numpy as np
import time

class BBBreakoutStrategy(BaseStrategy):
    def __init__(self, ai_model=None):
        super().__init__("BB_BREAKOUT")
        self.last_signal_time = None
        self.throttle_timer = 0
        self.bb_length = 20
        self.bb_std = 2.0

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        signal_payload = None
        action_msg = "[SQUEEZE_SCAN]"
        
        df = df_m5.copy()
        bbands = ta.bbands(df['close'], length=self.bb_length, std=self.bb_std)
        if bbands is None or len(bbands) < self.bb_length:
            return {'payload': None, 'ui': "[BB_INIT]"}
            
        bbu_col = [c for c in bbands.columns if c.startswith('BBU')][0]
        bbm_col = [c for c in bbands.columns if c.startswith('BBM')][0]
        bbl_col = [c for c in bbands.columns if c.startswith('BBL')][0]
        
        df['BBU'] = bbands[bbu_col]
        df['BBM'] = bbands[bbm_col]
        df['BBL'] = bbands[bbl_col]
        
        # Calculate strict Bandwidth: (Upper - Lower) / Middle
        df['BandWidth'] = (df['BBU'] - df['BBL']) / df['BBM']
        
        # Squeeze detection (Is bandwidth in the lowest 10% of last 100 periods?)
        df['BW_Min_100'] = df['BandWidth'].rolling(100).min()
        df['BW_Max_100'] = df['BandWidth'].rolling(100).max()
        df['BW_Percentile'] = (df['BandWidth'] - df['BW_Min_100']) / (df['BW_Max_100'] - df['BW_Min_100'] + 0.00001)
        
        # Volume Spike detection
        df['Vol_MA_20'] = df['tick_volume'].rolling(20).mean()
        df['Vol_Surge'] = df['tick_volume'] / (df['Vol_MA_20'] + 0.0001)
        
        latest = df.iloc[-1]
        last_time = df.iloc[-1]['time']
        
        bw_pct = latest.get('BW_Percentile', 1.0)
        is_squeezed = bw_pct < 0.15  # Squeeze Active: Lower 15% percentile of bandwidth
        
        if is_squeezed:
            action_msg = "[SQUEEZE_ACTIVE]"
        else:
            action_msg = f"[BW]: {bw_pct*100:.0f}%"

        # Signal Logic
        if self.last_signal_time != last_time and (time.time() - self.throttle_timer > 3.0):
            r_pct = min(current_risk, 0.5) 
            sl_padding = atr * 1.5  # Wider SL for momentum breakout
            
            # If price breaks completely outside bollinger bands and volume is massively surging (2x average)
            vol_surge = latest.get('Vol_Surge', 0)
            
            if latest['close'] > latest['BBU'] and vol_surge > 2.0 and is_squeezed:
                self.throttle_timer = time.time()
                self.last_signal_time = last_time
                signal_payload = {
                    'signal': 'BUY', 
                    'sl': current_price - sl_padding, 
                    'risk_override': r_pct, 
                    'confidence': 0.90, 
                    'comment': "BB SQUEEZE BUY"
                }
                print(f"\n[BB] [BOLLINGER SQUEEZE BREAKOUT] (Volume {vol_surge:.1f}x) -> RIDING UP")
                
            elif latest['close'] < latest['BBL'] and vol_surge > 2.0 and is_squeezed:
                self.throttle_timer = time.time()
                self.last_signal_time = last_time
                signal_payload = {
                    'signal': 'SELL', 
                    'sl': current_price + sl_padding, 
                    'risk_override': r_pct, 
                    'confidence': 0.90, 
                    'comment': "BB SQUEEZE SELL"
                }
                print(f"\n[BB] [BOLLINGER SQUEEZE BREAKOUT] (Volume {vol_surge:.1f}x) -> RIDING DOWN")

        return {'payload': signal_payload, 'ui': action_msg}
