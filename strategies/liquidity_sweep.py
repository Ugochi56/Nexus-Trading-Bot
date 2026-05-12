import pandas as pd
import numpy as np
from strategies.base import BaseStrategy

class LiquiditySweepStrategy(BaseStrategy):
    def __init__(self, ai_model):
        super().__init__(ai_model)
        self.lookback = 100
        self.proximity_threshold_pct = 0.0003 # 0.03% difference for extremely tight liquidity pools
        self.active_sweeps = set()

    def find_fractals(self, df):
        """Identifies Swing Highs and Swing Lows (Fractals) over a 5-candle window."""
        highs = df['high'].values
        lows = df['low'].values
        
        swing_highs = []
        swing_lows = []
        
        # Start from 2 and end at len-2 to check 2 candles before and after
        for i in range(2, len(df)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])
                
        return swing_highs, swing_lows

    def find_liquidity_pools(self, swing_highs, swing_lows):
        """Groups nearby swing points into Relative Equal Highs/Lows."""
        eqh_zones = []
        eql_zones = []
        
        # Group highs (BSL)
        used_h = set()
        for i, h1 in enumerate(swing_highs):
            if i in used_h: continue
            cluster = [h1]
            for j, h2 in enumerate(swing_highs[i+1:], start=i+1):
                if j in used_h: continue
                if abs(h1 - h2) / h1 <= self.proximity_threshold_pct:
                    cluster.append(h2)
                    used_h.add(j)
            # Must have at least 2 touches to be a liquidity pool
            if len(cluster) >= 2:
                avg_price = sum(cluster) / len(cluster)
                eqh_zones.append(avg_price)
                
        # Group lows (SSL)
        used_l = set()
        for i, l1 in enumerate(swing_lows):
            if i in used_l: continue
            cluster = [l1]
            for j, l2 in enumerate(swing_lows[i+1:], start=i+1):
                if j in used_l: continue
                if abs(l1 - l2) / l1 <= self.proximity_threshold_pct:
                    cluster.append(l2)
                    used_l.add(j)
            if len(cluster) >= 2:
                avg_price = sum(cluster) / len(cluster)
                eql_zones.append(avg_price)
                
        return eqh_zones, eql_zones

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, ai_mode=True):
        if len(df_m5) < self.lookback + 5:
            return {'ui': ''}

        # Analyze the last 100 M5 candles
        df_scan = df_m5.iloc[-self.lookback:]
        swing_highs, swing_lows = self.find_fractals(df_scan)
        eqh_zones, eql_zones = self.find_liquidity_pools(swing_highs, swing_lows)
        
        ui_msg = ""
        payload = None

        current_high = df_m5['high'].iloc[-1]
        current_low = df_m5['low'].iloc[-1]
        current_close = df_m5['close'].iloc[-1]

        sweep_found = False

        # 1. BSL Sweep (Judas Swing Up)
        # Price spikes above EQH (grabbing buy-stops), but immediately rejects and closes below it
        for eqh in eqh_zones:
            if current_high > eqh and current_close < eqh:
                sweep_found = True
                ui_msg = f"[BSL SWEPT: {eqh:.2f}]"
                
                # Check AI validation
                features = np.zeros((1, 20))
                confidence = self.get_ai_confidence(features) if ai_mode else 0.80
                
                if confidence >= 0.65:
                    payload = {
                        'signal': 'SELL',
                        'sl': current_high + (atr * 0.2), # SL just above the sweeping wick
                        'limit_price': eqh, # Limit order precisely at the swept liquidity line
                        'confidence': confidence,
                        'comment': f"LIQ_BSL_SWEEP"
                    }
                break

        # 2. SSL Sweep (Judas Swing Down)
        # Price spikes below EQL (grabbing sell-stops), but immediately rejects and closes above it
        if not sweep_found:
            for eql in eql_zones:
                if current_low < eql and current_close > eql:
                    sweep_found = True
                    ui_msg = f"[SSL SWEPT: {eql:.2f}]"
                    
                    # Check AI validation
                    features = np.zeros((1, 20))
                    confidence = self.get_ai_confidence(features) if ai_mode else 0.80
                    
                    if confidence >= 0.65:
                        payload = {
                            'signal': 'BUY',
                            'sl': current_low - (atr * 0.2), # SL just below the sweeping wick
                            'limit_price': eql, # Limit order precisely at the swept liquidity line
                            'confidence': confidence,
                            'comment': f"LIQ_SSL_SWEEP"
                        }
                    break
                
        # If no active sweep is occurring, display the nearest liquidity pools for the HUD
        if not sweep_found and eqh_zones and eql_zones:
            closest_eqh = min(eqh_zones, key=lambda x: abs(x - current_price))
            closest_eql = min(eql_zones, key=lambda x: abs(x - current_price))
            ui_msg = f"[LIQ BSL:{closest_eqh:.2f} SSL:{closest_eql:.2f}]"
                
        return {'ui': ui_msg, 'payload': payload}
