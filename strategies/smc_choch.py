import sys
from strategies.base import BaseStrategy
from core.config import *
from core.indicators import map_market_structure, calculate_poc
import time

class SMCChochStrategy(BaseStrategy):
    def __init__(self, ai_model=None):
        super().__init__("SMC_CHOCH")
        self.trend_model = ai_model
        self.last_traded_choch_time = None
        self.throttle_timer = 0
        self.structural_lookback = 40

    def get_trend_ai_permission(self, df_m5):
        if not USE_AI_FILTER or self.trend_model is None:
            return 'SKIP_CHECK', 0.0
        try:
            import pandas_ta as ta
            import numpy as np
            df = df_m5.copy()
            if 'EMA_50' not in df.columns:
                df['EMA_50'] = df.ta.ema(length=50)
                df['EMA_200'] = df.ta.ema(length=200)
                df['RSI'] = df.ta.rsi(length=14)
                df['ATR'] = df.ta.atr(length=14)
                adx = df.ta.adx(length=14)
                df['ADX'] = adx['ADX_14'] if adx is not None else 0

            latest = df.iloc[-2:].copy()
            latest['Dist_EMA_50'] = (latest['close'] - latest['EMA_50']) / latest['close']
            latest['Dist_EMA_200'] = (latest['close'] - latest['EMA_200']) / latest['close']
            
            # Simplified feature extraction since we don't have H1/H4 here in the signature
            # We will use zeros for the missing H1/H4 features to satisfy the model shape if needed, 
            # or rely on the base 7-feature model depending on what trend_model expects.
            # In london_breakout we used 7 features: 'Dist_EMA_50', 'Dist_EMA_200', 'Dist_H1', 'RSI', 'RSI_Zone', 'Rel_Volatility', 'ADX'
            # Let's rebuild the proxy H1
            df['EMA_H1_Proxy'] = df.ta.ema(length=600)
            latest['Dist_H1'] = (latest['close'] - df['EMA_H1_Proxy'].iloc[-1]) / latest['close']
            latest['Candle_Size'] = (latest['high'] - latest['low'])
            latest['Rel_Volatility'] = latest['Candle_Size'] / latest['ATR']
            latest['RSI_Zone'] = 1
            latest.loc[latest['RSI'] > 70, 'RSI_Zone'] = 2
            latest.loc[latest['RSI'] < 30, 'RSI_Zone'] = 0
            # H1/H4 proxies using M5 EMA for model shape consistency
            latest['H1_RSI'] = latest['RSI']  # Proxy: use M5 RSI
            latest['H4_ADX'] = latest['ADX']  # Proxy: use M5 ADX

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

    def find_origin_ob(self, df, start_idx, end_idx, ob_type):
        """
        Finds the Order Block (last opposite candle) in the leg that caused the CHOCH.
        start_idx: The index of the High/Low that started the CHOCH leg.
        end_idx: The index of the candle that broke structure.
        """
        if start_idx >= end_idx: return None
        
        atr = df['high'].iloc[-14:] - df['low'].iloc[-14:]
        avg_candle_size = atr.mean()
        
        if ob_type == 'SELL':
            # Looking for the last Bullish candle before the bearish drop
            for i in range(start_idx, end_idx):
                if df.iloc[i]['close'] > df.iloc[i]['open']:
                    zone_high = max(df.iloc[i]['high'], df.iloc[i-1]['high']) if i > 0 else df.iloc[i]['high']
                    zone_low = min(df.iloc[i]['low'], df.iloc[i-1]['low']) if i > 0 else df.iloc[i]['low']
                    return {'top': zone_high, 'bottom': zone_low}
                    
        elif ob_type == 'BUY':
            # Looking for the last Bearish candle before the bullish surge
            for i in range(start_idx, end_idx):
                if df.iloc[i]['close'] < df.iloc[i]['open']:
                    zone_high = max(df.iloc[i]['high'], df.iloc[i-1]['high']) if i > 0 else df.iloc[i]['high']
                    zone_low = min(df.iloc[i]['low'], df.iloc[i-1]['low']) if i > 0 else df.iloc[i]['low']
                    return {'top': zone_high, 'bottom': zone_low}
        return None

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, atr, **kwargs):
        signal_payload = None
        action_msg = "[SCANNING]"
        
        # 1. Map Structural Pivots
        structure = map_market_structure(df_m5, lookback=2)
        if len(structure) < 3:
            return {'payload': None, 'ui': "[AWAITING STRUCTURE]"}
            
        current_candle = df_m5.iloc[-1]
        current_idx = len(df_m5) - 1
        
        # We need the most recent Pivot High and Pivot Low
        recent_highs = [p for p in structure if p['type'] == 'HIGH']
        recent_lows = [p for p in structure if p['type'] == 'LOW']
        
        if not recent_highs or not recent_lows:
            return {'payload': None, 'ui': "[MAPPING PIVOTS]"}
            
        last_high = recent_highs[-1]
        last_low = recent_lows[-1]
        
        action_msg = f"[STRUCT H:{last_high['price']:.1f} L:{last_low['price']:.1f}]"

        # Throttle to prevent spam
        if time.time() - self.throttle_timer < 3.0:
            return {'payload': None, 'ui': action_msg}

        # 2. Detect Bearish CHOCH
        # Condition: Last Pivot was a High. Price drops and closes below the preceding Pivot Low.
        if structure[-1]['type'] == 'HIGH':
            preceding_lows = [p for p in structure if p['type'] == 'LOW' and p['idx'] < last_high['idx']]
            if preceding_lows:
                target_low = preceding_lows[-1]
                if current_candle['close'] < target_low['price'] and current_candle['time'] != self.last_traded_choch_time:
                    # Bearish CHOCH Confirmed
                    ai_verdict, ai_conf = self.get_trend_ai_permission(df_m5)
                    if ai_verdict == 'SELL':
                        ob = self.find_origin_ob(df_m5, last_high['idx'], current_idx, 'SELL')
                        if ob:
                            print(f"\n[SMC] [BEARISH CHOCH CONFIRMED] AI:{ai_conf:.2f}")
                            print(f"      Sweep at {last_high['price']:.2f}, Structural Break at {target_low['price']:.2f}")
                            print(f"      Placing Limit Order at Origin OB: {ob['bottom']:.2f}-{ob['top']:.2f}")
                            self.last_traded_choch_time = current_candle['time']
                            self.throttle_timer = time.time()
                            
                            sl = ob['top'] + (atr * getattr(sys.modules['core.config'], 'SL_ATR_MULTIPLIER', 1.5))
                            limit_entry = ob['bottom']
                            signal_payload = {'signal': 'SELL', 'sl': sl, 'limit_price': limit_entry, 'tp_price': target_low['price'], 'confidence': ai_conf, 'comment': f"CHOCH_DN:{ai_conf:.2f}"}
                            return {'payload': signal_payload, 'ui': "[BEARISH CHOCH]"}

        # 3. Detect Bullish CHOCH
        # Condition: Last Pivot was a Low. Price surges and closes above the preceding Pivot High.
        if structure[-1]['type'] == 'LOW':
            preceding_highs = [p for p in structure if p['type'] == 'HIGH' and p['idx'] < last_low['idx']]
            if preceding_highs:
                target_high = preceding_highs[-1]
                if current_candle['close'] > target_high['price'] and current_candle['time'] != self.last_traded_choch_time:
                    # Bullish CHOCH Confirmed
                    ai_verdict, ai_conf = self.get_trend_ai_permission(df_m5)
                    if ai_verdict == 'BUY':
                        ob = self.find_origin_ob(df_m5, last_low['idx'], current_idx, 'BUY')
                        if ob:
                            print(f"\n[SMC] [BULLISH CHOCH CONFIRMED] AI:{ai_conf:.2f}")
                            print(f"      Sweep at {last_low['price']:.2f}, Structural Break at {target_high['price']:.2f}")
                            print(f"      Placing Limit Order at Origin OB: {ob['bottom']:.2f}-{ob['top']:.2f}")
                            self.last_traded_choch_time = current_candle['time']
                            self.throttle_timer = time.time()
                            
                            sl = ob['bottom'] - (atr * getattr(sys.modules['core.config'], 'SL_ATR_MULTIPLIER', 1.5))
                            limit_entry = ob['top']
                            signal_payload = {'signal': 'BUY', 'sl': sl, 'limit_price': limit_entry, 'tp_price': target_high['price'], 'confidence': ai_conf, 'comment': f"CHOCH_UP:{ai_conf:.2f}"}
                            return {'payload': signal_payload, 'ui': "[BULLISH CHOCH]"}

        return {'payload': signal_payload, 'ui': action_msg}
