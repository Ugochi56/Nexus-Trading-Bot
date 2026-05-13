import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from core.config import *

class LiquiditySweepStrategy(BaseStrategy):
    def __init__(self, ai_model):
        super().__init__("LIQUIDITY")
        self.trend_model = ai_model
        self.lookback = 100
        self.proximity_threshold_pct = 0.0003 # 0.03% difference for extremely tight liquidity pools
        self.active_sweeps = set()
        self.last_traded_sweep_time = None

    def get_trend_ai_permission(self, df_m5, df_h1, df_h4):
        if not USE_AI_FILTER or self.trend_model is None:
            return 'SKIP_CHECK', 0.0
        try:
            df = df_m5.copy()
            if 'EMA_50' not in df.columns:
                import pandas_ta as ta
                df['EMA_50'] = df.ta.ema(length=50)
                df['EMA_200'] = df.ta.ema(length=200)
                df['RSI'] = df.ta.rsi(length=14)
                df['ATR'] = df.ta.atr(length=14)
                adx = df.ta.adx(length=14)
                if adx is not None: df['ADX'] = adx['ADX_14']
                else: df['ADX'] = 0

            latest = df.iloc[-2:].copy()
            latest['Dist_EMA_50'] = (latest['close'] - latest['EMA_50']) / latest['close']
            latest['Dist_EMA_200'] = (latest['close'] - latest['EMA_200']) / latest['close']
            latest['Candle_Size'] = (latest['high'] - latest['low'])
            latest['Rel_Volatility'] = latest['Candle_Size'] / latest['ATR']
            latest['RSI_Zone'] = 1
            latest.loc[latest['RSI'] > 70, 'RSI_Zone'] = 2
            latest.loc[latest['RSI'] < 30, 'RSI_Zone'] = 0

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

            feature_cols = ['Dist_EMA_50', 'Dist_EMA_200', 'Dist_H1', 'RSI', 'RSI_Zone', 'Rel_Volatility', 'ADX', 'H1_RSI', 'H4_ADX']
            X_live = latest[feature_cols].iloc[[-1]]

            probs = self.trend_model.predict_proba(X_live)
            prob_down = probs[0][0]
            prob_up = probs[0][1]

            if prob_up >= AI_CONFIDENCE_THRESHOLD: return 'BUY', prob_up
            elif prob_down >= AI_CONFIDENCE_THRESHOLD: return 'SELL', prob_down
            else: return 'UNCERTAIN', max(prob_up, prob_down)
        except Exception as e:
            return 'SKIP_CHECK', 0.0

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
        current_time = df_m5['time'].iloc[-1]

        sweep_found = False

        # 1. BSL Sweep (Judas Swing Up)
        # Price spikes above EQH (grabbing buy-stops), but immediately rejects and closes below it
        for eqh in eqh_zones:
            if current_high > eqh and current_close < eqh:
                sweep_found = True
                ui_msg = f"[BSL SWEPT: {eqh:.2f}]"
                
                # Check AI validation
                ai_verdict, confidence = self.get_trend_ai_permission(df_m5, df_h1, df_h4) if ai_mode else ('SELL', 0.80)
                
                if ai_verdict == 'SELL' and confidence >= 0.65 and current_time != self.last_traded_sweep_time:
                    payload = {
                        'signal': 'SELL',
                        'sl': current_high + (atr * 0.2), # SL just above the sweeping wick
                        'limit_price': eqh, # Limit order precisely at the swept liquidity line
                        'confidence': confidence,
                        'comment': f"LIQ_BSL_SWEEP"
                    }
                    self.last_traded_sweep_time = current_time
                break

        # 2. SSL Sweep (Judas Swing Down)
        # Price spikes below EQL (grabbing sell-stops), but immediately rejects and closes above it
        if not sweep_found:
            for eql in eql_zones:
                if current_low < eql and current_close > eql:
                    sweep_found = True
                    ui_msg = f"[SSL SWEPT: {eql:.2f}]"
                    
                    # Check AI validation
                    ai_verdict, confidence = self.get_trend_ai_permission(df_m5, df_h1, df_h4) if ai_mode else ('BUY', 0.80)
                    
                    if ai_verdict == 'BUY' and confidence >= 0.65 and current_time != self.last_traded_sweep_time:
                        payload = {
                            'signal': 'BUY',
                            'sl': current_low - (atr * 0.2), # SL just below the sweeping wick
                            'limit_price': eql, # Limit order precisely at the swept liquidity line
                            'confidence': confidence,
                            'comment': f"LIQ_SSL_SWEEP"
                        }
                        self.last_traded_sweep_time = current_time
                    break
                
        # If no active sweep is occurring, display the nearest liquidity pools for the HUD
        if not sweep_found and eqh_zones and eql_zones:
            closest_eqh = min(eqh_zones, key=lambda x: abs(x - current_price))
            closest_eql = min(eql_zones, key=lambda x: abs(x - current_price))
            ui_msg = f"[LIQ B:{closest_eqh:.1f} S:{closest_eql:.1f}]"
                
        return {'ui': ui_msg, 'payload': payload}
